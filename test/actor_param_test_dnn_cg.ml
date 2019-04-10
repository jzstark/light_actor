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
  mutable nn     : network;
}

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
    ~batch:(Batch.Mini 100) ~learning_rate:(Learning_Rate.Const 0.0001) 0.1

(* Utilities *)

let make_task nn = {
  state = None;
  nn;
}

let delta_nn nn0 nn1 =
  let par0 = Graph.mkpar nn0 in
  let par1 = Graph.mkpar nn1 in
  let delta = Owl_utils.aarr_map2 (fun a0 a1 -> Maths.(a0 - a1)) par0 par1 in
  Graph.update nn0 delta

let get_next_batch () =
  let x, _, y = Dataset.load_mnist_train_data_arr () in
  (* let x, y = Dataset.draw_samples_cifar x y 500 in *)
  x, y


(* for debugging only .. *)
(* let print_conv_weight nn =
  (Graph.mkpar nn).(2).(0)
  |> unpack_arr
  |> Dense.Ndarray.S.get_slice [[0];[0];[];[]]
  |> Dense.Ndarray.S.print *)


module Impl = struct

  type key = string

  type value = task

  type model = (key, value) Hashtbl.t

  let start_t = ref 0 (* used in stop function #2 *)

  let model : model =
    let nn = make_network () in
    Graph.init nn;
    let htbl = Hashtbl.create 10 in
    Hashtbl.add htbl "a" (make_task (Graph.copy nn));
    htbl

  let get keys =
    Array.map (Hashtbl.find model) keys

  let set kv_pairs =
    Array.iter (fun (key, value) ->
      Hashtbl.replace model key value
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
    Array.map (fun (k, v) ->
      Actor_log.info "push: %s, %s" k (Graph.get_network_name v.nn);
      let ps_nn = Graph.copy v.nn in
      let x, y = get_next_batch () in
      let x = pack x in
      let y = pack y in
      let state = match v.state with
        | Some state -> CGCompiler.train ~state ~params v.nn x y
        | None       -> CGCompiler.train ~params v.nn x y
      in
      Checkpoint.(state.current_batch <- 1);
      Checkpoint.(state.stop <- false);
      v.state <- Some state;
      (* return gradient instead of weight *)
      delta_nn v.nn ps_nn;
      (k, v)
    ) kv_pairs

  (* on server *)
  let pull kv_pairs =
    Array.map (fun (k, v) ->
      Actor_log.info "push: %s, %s" k (Graph.get_network_name v.nn);
      let u = (get [|k|]).(0) in
      let par0 = Graph.mkpar u.nn in
      let par1 = Graph.mkpar v.nn in
      Owl_utils.aarr_map2 (fun a0 a1 ->
        Maths.(a0 + a1)
      ) par0 par1
      |> Graph.update v.nn;
      (k, v)
    ) kv_pairs

  let stop () =
    let v = (get [|"a"|]).(0) in
    match v.state with
    | Some state ->
        let len = Array.length state.loss in
        let loss = state.loss.(len - 1) |> unpack_flt in
        if (loss < 2.0) then true else false
    | None       -> false

end


include Actor_param_types.Make(Impl)


module M = Actor_param.Make (Actor_net_zmq) (Actor_sys_unix) (Impl)


let ip_of_uuid id =
  try
    (Unix.gethostbyname id).h_addr_list.(0)
    |> Unix.string_of_inet_addr
  with Not_found -> "127.0.0.1"

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
