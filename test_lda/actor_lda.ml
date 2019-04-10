module L = Owl_nlp_lda
module M = Owl_dense_matrix.S

open Owl_nlp

type twk = float array array

let iter = 100
let num_topic = 10

let aarr_add a b =
  let ma = M.of_arrays a in
  let mb = M.of_arrays b in
  M.add ma mb |> M.to_arrays


let aarr_sub a b =
  let ma = M.of_arrays a in
  let mb = M.of_arrays b in
  M.sub ma mb |> M.to_arrays

module Impl = struct

  type key = string (* a meaningless name...? *)

  type value = twk

  type model = (key, value) Hashtbl.t

  let model : model =
    let htbl = Hashtbl.create 5 in

    let doc = Corpus.build "test.txt" in
    let voc = Corpus.get_vocab doc |> Vocabulary.get_w2i in
    let m = L.init ~iter num_topic voc doc in
    let foo = L.get_twk m in

    Hashtbl.add htbl "twk" foo;
    htbl

  let get keys =
    Array.map (Hashtbl.find model) keys

  let set kv_pairs =
    Array.iter (fun (key, value) ->
      Hashtbl.replace model key value
    ) kv_pairs


  (* on server *)
  let schd nodes =
    Actor_log.info "fuck";
    Array.map (fun node ->
      Actor_log.info "node: %s schd" node;
      let key = "twk" in
      let value = (get [|key|]).(0) in
      let tasks = [|(key, value)|] in
      (node, tasks)
    ) nodes


  (* on worker *)
  let push kv_pairs =
    Actor_log.info "shit!";
    Array.map (fun (k, v) ->
      Actor_log.info "push: %s" k;

      let doc = Corpus.build "test.txt" in
      let voc = Corpus.get_vocab doc |> Vocabulary.get_w2i in
      let m = L.init ~iter num_topic voc doc in

      M.(of_arrays (L.get_twk m) |> print);
      L.set_twk m v;
      M.of_arrays (L.get_twk m) |> M.print;

      L.(train SimpleLDA m);

      let u = L.get_twk m in
      let delta = aarr_sub u v in
      (k, delta)
    ) kv_pairs


  (* on server *)
  let pull kv_pairs =
    Array.map (fun (k, v) ->
      Actor_log.info "pull: %s" k;
      let u = (get [|k|]).(0) in
      let value = aarr_add u v in
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

include Actor_param_types.Make(Impl)

module A = Actor_param.Make (Actor_net_zmq) (Actor_sys_unix) (Impl)

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
  (* Actor_book.add book "w1" "" true (-1); *)
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
  Lwt_main.run (A.init context)


let _ =
  main Sys.argv
