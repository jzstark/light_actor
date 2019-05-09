(*
 * Light Actor - Parallel & Distributed Engine of Owl System
 * Copyright (c) 2016-2019 Liang Wang <liang.wang@cl.cam.ac.uk>
 *)


let is_ready book =
  let ready = ref true in
  Hashtbl.iter (fun _ n ->
    if Actor_book.(String.length n.addr = 0) then
      ready := false
  ) book;
  !ready


let arr_to_htbl arr =
  let len = Array.length arr in
  let htbl = Hashtbl.create len in
  Array.iter (fun k ->
    Hashtbl.add htbl k k
  ) arr;
  htbl


let htbl_to_arr htbl =
  Hashtbl.fold (fun k v acc ->
    Array.append acc [| (k,v) |]
  ) htbl [||]


let range a b =
    let rec aux a b =
      if a > b then [] else a :: aux (a+1) b  in
    if a > b then List.rev (aux b a) else aux a b;;


let rand_select list n =
    let rec extract acc n = function
      | [] -> raise Not_found
      | h :: t -> if n = 0 then (h, acc @ t) else extract (h::acc) (n-1) t
    in
    let extract_rand list len =
      extract [] (Random.int len) list
    in
    let rec aux n acc list len =
      if n = 0 then acc else
        let picked, rest = extract_rand list len in
        aux (n-1) (picked :: acc) rest (len-1)
    in
    let len = List.length list in
    aux (min n len) [] list len;;


(* draw n different random numbers from the set 0..m *)
let lotto_select n m =
  rand_select (range 0 m) n |> Array.of_list
