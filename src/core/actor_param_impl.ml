(*
 * Light Actor - Parallel & Distributed Engine of Owl System
 * Copyright (c) 2016-2019 Liang Wang <liang.wang@cl.cam.ac.uk>
 *)


module type Sig = sig

  type model   (* abstract model definition *)

  type key     (* key for refering a value in the model *)

  type value   (* value type of the model *)

  val get : key array -> value array

  val set : (key * value) array -> unit

  val schd : string array -> (string * (key * value) array) array

  val push : (key * value) array -> (key * value) array

  val pull : (key * value) array -> (key * value) array

  val stop : (key * value) array -> bool

end
