(*
 * Light Actor - Parallel & Distributed Engine of Owl System
 * Copyright (c) 2016-2019 Liang Wang <liang.wang@cl.cam.ac.uk>
 *)


module Make
  (Impl : Actor_param_impl.Sig)
  = struct

  type operation =
    (* General message types *)
    | Reg_Req
    | Reg_Rep
    | Heartbeat of int
    | Exit      of int
    (* Model parallel message types *)
    | PS_Get    of Impl.key array
    | PS_Set    of (Impl.key * Impl.value) array
    | PS_Schd   of (Impl.key * Impl.value) array
    | PS_Push   of (Impl.key * Impl.value) array


  type message = {
    mutable uuid      : string;
    mutable addr      : string;
    mutable operation : operation;
  }


  type param_context = {
    mutable my_uuid      : string;
    mutable my_addr      : string;
    mutable server_uuid  : string;
    mutable server_addr  : string;
    mutable book         : Actor_book.t;
    mutable barrier_args : int * int option (* staleness, sampling size *)
  }


  let encode_message uuid addr operation =
    let m = { uuid; addr; operation } in
    Marshal.to_string m []


  let decode_message data : message
    = Marshal.from_string data 0

end
