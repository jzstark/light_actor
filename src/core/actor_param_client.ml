(*
 * Light Actor - Parallel & Distributed Engine of Owl System
 * Copyright (c) 2016-2019 Liang Wang <liang.wang@cl.cam.ac.uk>
 *)

module Make
  (Net  : Actor_net.Sig)
  (Sys  : Actor_sys.Sig)
  (Impl : Actor_param_impl.Sig)
  = struct

  include Actor_param_types.Make(Impl)


  let register s_addr c_uuid c_addr =
    Actor_log.debug ">>> %s Reg_Req" s_addr;
    let s = encode_message c_uuid c_addr Reg_Req in
    Net.send s_addr s


  (* let heartbeat s_addr c_uuid c_addr =
    let rec loop i =
      try%lwt (
        let%lwt () = Sys.sleep 10. in
        Actor_log.debug ">>> %s Heartbeat #%i" s_addr i;
        let s = encode_message c_uuid c_addr (Heartbeat i) in
        let%lwt () = Net.send s_addr s in
        loop (i + 1)
      ) with Lwt.Canceled ->
        Lwt.return ()
    in
    loop 0 *)
    (* try%lwt loop 0 with Lwt.Canceled -> Lwt.return () *)
    (* match Lwt.state main_thd with
    | Lwt.Return _ -> Actor_log.info "return"; Lwt.return ()
    | Lwt.Sleep  -> Actor_log.info "sleep"; loop (i + 1)
    | Lwt.Fail _ -> Actor_log.info "fail"; loop (i + 1) *)


  let heartbeat s_addr c_uuid c_addr =
    let rec loop i =
      let%lwt () = Sys.sleep 10. in
      Actor_log.debug ">>> %s Heartbeat #%i" s_addr i;
      let s = encode_message c_uuid c_addr (Heartbeat i) in
      let%lwt () = Net.send s_addr s in
      loop (i + 1)
    in
    loop 0


  let process context data =
    let m = decode_message data in
    let my_uuid = context.my_uuid in
    let my_addr = context.my_addr in

    match m.operation with
    | Reg_Rep -> (
        Actor_log.debug "<<< %s Reg_Rep" m.uuid;
        Lwt.return ()
      )
    | Exit code -> (
        Actor_log.debug "<<< %s Exit %i" m.uuid code;
        Lwt.fail (Lwt.Canceled)
      )
    | PS_Schd tasks -> (
        Actor_log.debug "<<< %s PS_Schd" m.uuid;
        let updates = Impl.push tasks in
        let s = encode_message my_uuid my_addr (PS_Push updates) in
        Net.send context.server_addr s
      )
    | _ -> (
        Actor_log.error "unknown message type";
        Lwt.return ()
      )


  let init context =
    let%lwt () = Net.init () in

    (* register client to server *)
    let uuid = context.my_uuid in
    let addr = context.my_addr in
    let%lwt () = register context.server_addr uuid addr in

    (* start client service *)
    let thread_0 = Net.listen addr (process context) in
    let thread_1 = heartbeat context.server_addr uuid addr in

    (* Lwt.on_cancel thread_1 (fun () -> Actor_log.info "foo");
    Lwt.on_cancel thread_0 (fun () -> Actor_log.info "foo!");
    Lwt.on_termination thread_0 (fun () -> Actor_log.info "fool!"); *)

    let%lwt () = Lwt.pick [thread_0; thread_1] in

    (* clean up when client exits *)
    let%lwt () = Net.exit () in
    Lwt.return ()


end
