open Owl
open Owl.Neural.S
open Graph
open Owl_algodiff.S
open Owl_optimise.S

module G = Owl.Neural.S.Graph

type task = {
  mutable state  : Checkpoint.state option;
  mutable params : Params.typ;
  mutable nn     : G.network;
  mutable data_x : t;
  mutable data_y : t;
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

(* Utilities *)

let make_task params nn data_x data_y = {
  state = None;
  params;
  nn;
  data_x;
  data_y;
}

let delta_model model0 model1 =
  let par0 = G.mkpar model0 in
  let par1 = G.mkpar model1 in
  let delta = Owl_utils.aarr_map2 (fun a0 a1 -> Maths.(a0 - a1)) par0 par1 in
  G.update model0 delta


module Impl = struct

  type model = (string, task) Hashtbl.t

  type key = string

  type value = task

  let model : model =
    Hashtbl.create 128


  let get keys =
    Array.map (Hashtbl.find model) keys


  let set kv_pairs =
    Array.iter (fun (key, value) ->
      Hashtbl.replace model key value
    ) kv_pairs


  let schd nodes =
    Array.map (fun node ->
      let key = Random.int 1000 |> string_of_int in
      let params = Params.config ~batch:(Batch.Sample 100) 1. in
      let nn = make_network () in
      let x, _, y = Owl.Dataset.load_cifar_train_data 1 in
      let value = make_task params nn (Arr x) (Arr y) in
      let tasks = [|(key, value)|] in
      (node, tasks)
    ) nodes


  let push kv_pairs =
    Array.map (fun (k, task) ->
      let old_model = G.copy task.nn in
      let params = task.params in
      let x = task.data_x in
      let y = task.data_y in
      let state = match task.state with
        | Some state -> G.(train_generic ~state ~params ~init_model:false task.nn x y)
        | None       -> G.(train_generic ~params ~init_model:false task.nn x y)
      in
      Checkpoint.(state.stop <- false);
      task.state <- Some state;
      delta_model task.nn old_model;
      (k, task)
    ) kv_pairs


  let pull updates = 
    Array.map (fun (key, task1) ->
      let task0 = (get [|key|]).(0) in
      let par0 = G.mkpar task0.nn in
      let par1 = G.mkpar task1.nn in
      Owl_utils.aarr_map2 (fun a0 a1 ->
        Maths.(a0 + a1)
      ) par0 par1
      |> G.update task0.nn;
      task1.nn <- task0.nn;
      (key, task1)
    ) updates

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

  (* define the participants *)
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
