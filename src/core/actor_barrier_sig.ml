(*
 * Light Actor - Parallel & Distributed Engine of Owl System
 * Copyright (c) 2016-2019 Liang Wang <liang.wang@cl.cam.ac.uk>
 *)

module type Sig = sig

  val pass : int -> int option -> Actor_book.t -> string array
  (* staleness, sampling size, book --> passed nodes *)

  val sync : Actor_book.t -> string -> unit

end
