open Owl
open Owl.Neural.S
open Graph
open Owl_algodiff.S
open Owl_optimise.S

module G = Owl.Neural.S.Graph

type task = {
  mutable state  : Checkpoint.state option;
  mutable nn     : G.network;
}

let make_network () =
  let nn =
    input [|32;32;3|]
    |> normalisation ~decay:0.9
    |> conv2d [|3;3;3;32|] [|1;1|] ~act_typ:Activation.Relu
    |> conv2d [|3;3;32;32|] [|1;1|] ~act_typ:Activation.Relu ~padding:VALID
    |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
    |> dropout 0.1
    |> conv2d [|3;3;32;64|] [|1;1|] ~act_typ:Activation.Relu
    |> conv2d [|3;3;64;64|] [|1;1|] ~act_typ:Activation.Relu ~padding:VALID
    |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
    |> dropout 0.1
    |> fully_connected 512 ~act_typ:Activation.Relu
    |> linear 10 ~act_typ:Activation.(Softmax 1)
    |> get_network
  in
  nn

let chkpt _state = ()
  (* if Checkpoint.(state.current_batch mod 1 = 0) then (
        Checkpoint.(state.stop <- true);
        (* Log.info "sync model with server" *)
      ) *)
let params = Params.config
      ~batch:(Batch.Sample 100) ~learning_rate:(Learning_Rate.Adagrad 0.005)
      ~checkpoint:(Checkpoint.Custom chkpt) ~stopping:(Stopping.Const 1e-6) 10.

(* Utilities *)

let make_task nn = {
  state = None;
  nn;
}

let delta_nn nn0 nn1 =
  let par0 = G.mkpar nn0 in
  let par1 = G.mkpar nn1 in
  let delta = Owl_utils.aarr_map2 (fun a0 a1 -> Maths.(a0 - a1)) par0 par1 in
  G.update nn0 delta

let get_next_batch () =
  let x, _, y = Dataset.load_cifar_train_data 1 in
  (* let col_num = (Owl.Dense.Ndarray.S.shape x).(0) in
  let a = Array.init col_num (fun i -> i) in
  let a = Owl_stats.sample a 200 |> Array.to_list in
  let _ = List.map (fun x -> Printf.fprintf stderr "%d " x) a in
  Owl_dense_ndarray.S.get_fancy [L a; R []; R []; R []] x,
  Owl_dense_matrix.S.get_fancy  [L a; R []] y *)
  Dataset.draw_samples_cifar x y 200


module Impl = struct

  type key = string

  type value = task

  type model = (key, value) Hashtbl.t

  let model : model =
    Hashtbl.create 10

  let get keys =
    Array.map (Hashtbl.find model) keys

  let set kv_pairs =
    Array.iter (fun (key, value) ->
      Hashtbl.replace model key value
    ) kv_pairs

  let schd nodes =
    let nn = make_network () in
    G.init nn;
    Array.map (fun node ->
      let key = Random.int 1000 |> string_of_int in
      let value = make_task (G.copy nn) in
      let tasks = [|(key, value)|] in
      set tasks;
      Actor_log.info "node: %s schd" node;
      (node, tasks)
    ) nodes


  (* on worker *)
  let push kv_pairs =
    (* should contain only one kvpair in this case *)
    Array.map (fun (k, v) ->
      Actor_log.info "push: %s, %s" k (G.get_network_name v.nn);
      Actor_log.info "before...";
      Dense.Ndarray.S.print (unpack_arr (G.mkpar v.nn).(2).(0));

      let ps_nn = G.copy v.nn in
      let x, y = get_next_batch () in
      (* Dense.Ndarray.S.print y; *)
      let state = match v.state with
        | Some state -> Actor_log.info "shit!"; G.(train_generic ~state ~params ~init_model:false v.nn (Arr x) (Arr y))
        | None       -> Actor_log.info "fuck!"; G.(train_generic ~params ~init_model:false ps_nn (Arr x) (Arr y))
      in
      Checkpoint.(state.stop <- false);
      v.state <- Some state;
      (* return grad instead of weight *)
      Actor_log.info "middle...";
      Dense.Ndarray.S.print (unpack_arr (G.mkpar ps_nn).(2).(0));
      (* delta_nn v.nn ps_nn;
      Actor_log.info "after...";
      Dense.Ndarray.S.print (unpack_arr (G.mkpar v.nn).(2).(0)); *)
      (* G.update v.nn state.gs; *)
      (k, v)
    ) kv_pairs


  let pull kv_pairs =
    Array.map (fun (k, v) ->
      Actor_log.info "push: %s, %s" k (G.get_network_name v.nn);
      let u = (get [|k|]).(0) in
      let par0 = G.mkpar u.nn in
      let par1 = G.mkpar v.nn in

      (* Actor_log.info "before...";
      Dense.Ndarray.S.print (unpack_arr (G.mkpar v.nn).(2).(0)); *)

      Owl_utils.aarr_map2 (fun a0 a1 ->
        Maths.(a0 + a1)
      ) par0 par1
      |> G.update v.nn;

      (* Actor_log.info "after...";
      Dense.Ndarray.S.print (unpack_arr (G.mkpar v.nn).(2).(0)); *)
      (k, v)
    ) kv_pairs

end


include Actor_param_types.Make(Impl)

module M = Actor_param.Make (Actor_net_zmq) (Actor_sys_unix) (Impl)

let main args =
  Actor_log.(set_level DEBUG);
  Random.self_init ();

  (* define server uuid and address *)
  let server_uuid = "server" in
  let server_addr = "tcp://127.0.0.1:5555" in

  (* define my own uuid and address *)
  let my_uuid = args.(1) in
  let my_addr =
    if my_uuid = server_uuid then
      server_addr
    else
      let port = string_of_int (6000 + Random.int 1000) in
      "tcp://127.0.0.1:" ^ port
  in

  (* define the participants -- hardcoded *)
  let book = Actor_book.make () in
  Actor_book.add book "w0" "" false (-1);
  Actor_book.add book "w1" "" false (-1);
  if my_uuid <> server_uuid then
    Actor_book.set_addr book my_uuid my_addr;

  (* define parameter server context *)
  let context = {
    my_uuid;
    my_addr;
    server_uuid;
    server_addr;
    book;
  }
  in

  (* start the event loop *)
  Lwt_main.run (M.init context)

let _ =
  main Sys.argv
