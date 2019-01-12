open Owl
open Neural.S
open Graph
open Neuron
open Owl_algodiff.S

module N = Dense.Ndarray.S

let make_network input_shape =
  input input_shape
  (* |> normalisation ~decay:0.9 *)
  |> lambda (fun x -> Maths.(x / F 256.))
  |> conv2d [|3;3;3;32|] [|1;1|] ~act_typ:Activation.Relu
  |> conv2d [|3;3;32;32|] [|1;1|] ~act_typ:Activation.Relu ~padding:VALID
  |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
  (* |> dropout 0.1
  |> conv2d [|3;3;32;64|] [|1;1|] ~act_typ:Activation.Relu
  |> conv2d [|3;3;64;64|] [|1;1|] ~act_typ:Activation.Relu ~padding:VALID
  |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
  |> dropout 0.1 *)
  |> fully_connected 512 ~act_typ:Activation.Relu
  |> linear 10 ~act_typ:Activation.(Softmax 1)
  |> get_network


let network = make_network [|32;32;3|] 
let _ = Graph.print network
let _ = Graph.init network


let _ = 
  Array.iter (fun node -> 
    match node.neuron with
    | Conv2D l -> 
      let sw = N.shape (unpack_arr l.w) in 
      let sb = N.shape (unpack_arr l.b) in
      l.w <- (Arr (N.mul_scalar (N.ones sw) 0.0005));
      l.b <- (Arr (N.zeros sb))
    | FullyConnected l -> 
      let sw = N.shape (unpack_arr l.w) in 
      let sb = N.shape (unpack_arr l.b) in
      (* l.w <- (Arr (N.zeros sw)); *)
      l.w <- (Arr (N.mul_scalar (N.ones sw) 0.0005));
      l.b <- (Arr (N.zeros sb))
    | Linear l -> 
      let sw = N.shape (unpack_arr l.w) in 
      let sb = N.shape (unpack_arr l.b) in
      (* l.w <- (Arr (N.zeros sw)); *)
      l.w <- (Arr (N.mul_scalar (N.ones sw) 0.0005));
      l.b <- (Arr (N.zeros sb))
    | _ -> ()
  ) network.topo


let get_data size = 
  let x, _, y = Dataset.load_cifar_train_data 1 in 
  let x = N.get_slice [[0; size - 1]; []; []; []] x in
  let y = N.get_slice [[0; size - 1]; []] y in
  x, y

let test network = 
  let x, _y = get_data 1 in 
  Owl_log.info "Testing network:... ";
  Graph.model network x |> Dense.Ndarray.S.print 

let x, y = get_data 200

let params = Params.config
    ~batch:(Batch.Sample 100) 
    ~learning_rate:(Learning_Rate.Adagrad 0.001)
    (* ~learning_rate:(Learning_Rate.Adam (0.1, 0.9, 0.999)) *)
    ~stopping:(Stopping.Const 1e-6) 1.


(* let _ = Owl_log.info "before training..."
let param = Graph.mkpar network;;
let _ = param.(2).(0) |> unpack_arr |> Dense.Ndarray.S.print *)

let _ = Graph.train ~params ~init_model:false network x y |> ignore

(*
let _ = Owl_log.info "after training..."
let param = Graph.mkpar network;;
let _ = param.(2).(0) |> unpack_arr |> Dense.Ndarray.S.print 
*)

let _ = test network