(*
 * Light Actor - Parallel & Distributed Engine of Owl System
 * Copyright (c) 2016-2019 Liang Wang <liang.wang@cl.cam.ac.uk>
 *)

 module type Param = sig

   val s : int

   val p : int option

 end

module type Sig = sig

  val pass : Actor_book.t -> string array

  val sync : Actor_book.t -> string -> unit

end
