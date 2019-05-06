(*
 * Light Actor - Parallel & Distributed Engine of Owl System
 * Copyright (c) 2016-2019 Liang Wang <liang.wang@cl.cam.ac.uk>
 *)

open Owl

module CPU_Engine = Owl_computation_cpu_engine.Make (Dense.Ndarray.S)
module CGCompiler = Owl_neural_compiler.Make (CPU_Engine)

open CGCompiler.Neural
open CGCompiler.Neural.Graph
open CGCompiler.Neural.Algodiff

type task = {
  mutable state  : Checkpoint.state option;
  mutable weight : t array array;
  mutable clock  : int;
}

let log_file = "psp_exp.csv"

let make_network () =
  input [|28;28;1|]
  |> normalisation ~decay:0.9
  |> conv2d [|5;5;1;32|] [|1;1|] ~act_typ:Activation.Relu
  |> max_pool2d [|2;2|] [|2;2|]
  |> dropout 0.1
  |> fully_connected 1024 ~act_typ:Activation.Relu
  |> linear 10 ~act_typ:Activation.(Softmax 1)
  |> get_network

let pack x = CGCompiler.Engine.pack_arr x |> Algodiff.pack_arr
let unpack x = Algodiff.unpack_arr x |> CGCompiler.Engine.unpack_arr


let chkpt _state = ()
let params = Params.config
    ~batch:(Batch.Mini 100) ~learning_rate:(Learning_Rate.Const 0.001) 0.05

(* Utilities *)

let eval_weight w =
  Owl_utils.aarr_map (fun x ->
    let y = Algodiff.unpack_arr x in
    CGCompiler.Engine.eval_arr [|y|];
    CGCompiler.Engine.unpack_arr y |> pack (* do NOT change this line!!! *)
  ) w


let make_task s w c =
  { state = s; weight = w; clock = c}


let delta_nn nn0 par1 =
  let par0 = Graph.mkpar nn0 |> eval_weight in
  let par1 = par1 |> eval_weight in
  Owl_utils.aarr_map2 (fun a0 a1 -> Maths.(a0 - a1)) par0 par1


let add_weight par0 par1 =
  let par0 = par0 |> eval_weight in
  let par1 = par1 |> eval_weight in
  Owl_utils.aarr_map2 (fun a0 a1 -> Maths.(a0 + a1)) par0 par1


let get_next_batch () =
  let x, _, y = Dataset.load_mnist_train_data_arr () in
  x, y


let test weight =
  let network = make_network () in
  Graph.update network weight;

  let imgs, _, labels = Dataset.load_mnist_test_data () in
  let m = Dense.Matrix.S.row_num imgs in
  let imgs = Dense.Ndarray.S.reshape imgs [|m;28;28;1|] in
  let eval = CGCompiler.model ~batch_size:100 network in

  let mat2num x = Dense.Matrix.S.of_array (
      x |> Dense.Matrix.Generic.max_rows
        |> Array.map (fun (_,_,num) -> float_of_int num)
    ) 1 m
  in

  let result = unpack (eval (pack imgs)) in
  let pred = mat2num result in
  let fact = mat2num labels in
  let accu = Dense.Matrix.S.(elt_equal pred fact |> sum') in
  (* Owl_log.info "Accuracy on test set: %f" (accu /. (float_of_int m)) *)
  accu /. (float_of_int m)


module Impl = struct

  type key = string

  type value = task

  type model = (key, value) Hashtbl.t

  let model : model =
    let nn = make_network () in
    Graph.init nn;
    let weight = Graph.(copy nn |> mkpar |> eval_weight) in
    let htbl = Hashtbl.create 10 in
    Hashtbl.add htbl "a" (make_task None weight 0);
    htbl


  let get keys =
    Array.map (Hashtbl.find model) keys


  let set kv_pairs =
    Array.iter (fun (k, v) ->
      (* append to log file *)
      let accuracy = test v.weight in
      let time = Unix.gettimeofday () in
      let oc = open_out_gen [Open_append; Open_creat] 0o666 log_file in
      Printf.fprintf oc "%.2f, %d, %.2f\n" time v.clock accuracy;
      close_out oc;
      (* update clock... for now. Later need to update so that, e.g., in BSP we only increase clock by one when multiple updates arrives at the same time. *)
      v.clock <- v.clock + 1;
      Hashtbl.replace model k v
    ) kv_pairs


  (* on server *)
  let schd nodes =
    Array.map (fun node ->
      Actor_log.info "node: %s schd" node;
      let key = "a" in
      let value = (get [|key|]).(0) in
      let tasks = [|(key, value)|] in
      (node, tasks)
    ) nodes


  (* on worker *)
  let push kv_pairs =
    Gc.compact ();

    Array.map (fun (k, v) ->
      Actor_log.info "push: %s" k;

      let nn = make_network () in
      Graph.init nn;
      Graph.update nn v.weight;

      let x, y = get_next_batch () in
      let x = pack x in
      let y = pack y in
      let _state = match v.state with
        | Some state -> CGCompiler.train ~state ~params ~init_model:false nn x y
        | None       -> CGCompiler.train ~params ~init_model:false nn x y
      in
      (* Checkpoint.(state.current_batch <- 1);
      Checkpoint.(state.stop <- false); *)

      let delta = delta_nn nn v.weight in
      let value = make_task None delta v.clock in
      (k, value)
    ) kv_pairs


  (* on server *)
  let pull kv_pairs =
    Gc.compact ();
    Array.map (fun (k, v) ->
      Actor_log.info "pull: %s" k;
      let u  = (get [|k|]).(0) in
      let u' = add_weight u.weight v.weight in
      let value = make_task v.state u' v.clock in
      (k, value)
    ) kv_pairs


  let stop () = false
    (* let v = (get [|"a"|]).(0) in
    match v.state with
    | Some state ->
        let len = Array.length state.loss in
        let loss = state.loss.(len - 1) |> unpack_flt in
        if (loss < 2.0) then true else false
    | None       -> false *)

end


include Actor_param_types.Make (Impl)


module SSP = struct
  let s = 3
  let p = None
end

module PBSP = struct
  let s = 0
  let p = Some 2
end

module PSSP = struct
  let s = 3
  let p = Some 2
end

module M_ASP = Actor_param.Make (Actor_net_zmq) (Actor_sys_unix) (Impl) (Actor_barrier_asp)

module M_BSP = Actor_param.Make (Actor_net_zmq) (Actor_sys_unix) (Impl) (Actor_barrier_bsp)

module M_SSP = Actor_param.Make (Actor_net_zmq) (Actor_sys_unix) (Impl) (Actor_barrier_ssp.Make (SSP))

module M_PBSP = Actor_param.Make (Actor_net_zmq) (Actor_sys_unix) (Impl) (Actor_barrier_ssp.Make (PSSP))

module M_PSSP = Actor_param.Make (Actor_net_zmq) (Actor_sys_unix) (Impl) (Actor_barrier_ssp.Make (PSSP))


let ip_of_uuid id =
  try
    (Unix.gethostbyname id).h_addr_list.(0)
    |> Unix.string_of_inet_addr
  with Not_found -> "127.0.0.1"


(* args: uuid, barrier *)
let main args =
  Actor_log.(set_level DEBUG);
  Random.self_init ();

  (* define server uuid and address *)
  let server_uuid = "server" in
  let server_port =
    try Unix.getenv "SERVER_PORT"
    with Not_found -> "5555" in
  let server_addr = "tcp://" ^ (ip_of_uuid  server_uuid) ^
                    ":" ^ server_port in

  (* define my own uuid and address *)
  let my_uuid = args.(1) in
  let my_addr =
    if my_uuid = server_uuid then
      server_addr
    else
      let port = try Unix.getenv "PORT"
        with Not_found ->
          string_of_int (6000 + Random.int 1000)
      in
      "tcp://" ^ (ip_of_uuid my_uuid) ^ ":" ^ port
  in

  let book = Actor_book.make () in
  Actor_book.add book "w0" "" true (-1);
  Actor_book.add book "w1" "" true (-1);
  (* Actor_book.add book "w2" "" true (-1);
  Actor_book.add book "w3" "" true (-1);
  Actor_book.add book "w4" "" true (-1);
  Actor_book.add book "w5" "" true (-1); *)
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
  let barrier = args.(2) in
  if (barrier = "bsp") then (
    Lwt_main.run (M_BSP.init context)
  ) else if (barrier = "asp") then (
    Lwt_main.run (M_ASP.init context)
  ) else if (barrier = "ssp") then (
    Lwt_main.run (M_SSP.init context)
  ) else if (barrier = "pbsp") then (
    Lwt_main.run (M_PBSP.init context)
  ) else if (barrier = "pbsp") then (
    Lwt_main.run (M_PBSP.init context)
  )

let _ =
  main Sys.argv
