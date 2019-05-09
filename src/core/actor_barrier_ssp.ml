(*
 * Light Actor - Parallel & Distributed Engine of Owl System
 * Copyright (c) 2016-2019 Liang Wang <liang.wang@cl.cam.ac.uk>
 *)

open Actor_book


let sample nodes_kv p =
  let n = Array.length nodes_kv in
  let idx = Actor_param_utils.lotto_select p (n - 1) in
  Array.map (Array.get nodes_kv) idx


(* module Make (P : Actor_barrier_sig.Param) = struct *)

  let pass s p book =

    let nodes_kv = Actor_param_utils.htbl_to_arr book in
    let nodes_kv = match p with
    | Some p -> sample nodes_kv p
    | None   -> nodes_kv
    in

    let fastest = Array.fold_left (fun acc (_, node) ->
      max node.step acc
    ) min_int nodes_kv
    in

    let slowest = Array.fold_left (fun acc (_, node) ->
      min node.step acc
    ) max_int nodes_kv
    in

    let passed = ref [||] in
    Array.iter (fun (uuid, node) ->
      if (fastest - node.step <= s && node.step - slowest <= s
        && node.busy = false) then (
        node.busy <- true;
        passed := Array.append !passed [| uuid |]
      )
    ) nodes_kv;
    !passed


  let sync book uuid =
    let step = Actor_book.get_step book uuid in
    Actor_book.set_busy book uuid false;
    Actor_book.set_step book uuid (step + 1)

(* end *)
