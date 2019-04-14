(*
 * Light Actor - Parallel & Distributed Engine of Owl System
 * Copyright (c) 2016-2019 Liang Wang <liang.wang@cl.cam.ac.uk>
 *)

open Actor_book


let pass book =
  Hashtbl.fold (fun uuid node acc ->
    if node.busy = false then (
      node.busy <- true;
      Array.append acc [| uuid |]
    )
    else acc
  ) book [| |]


let sync book uuid =
  let step = Actor_book.get_step book uuid in
  Actor_book.set_busy book uuid false;
  Actor_book.set_step book uuid (step + 1)
