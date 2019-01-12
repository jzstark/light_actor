open Owl
open Neural.S
open Neural.S.Graph
open Owl_algodiff.S


let make_network input_shape =
  input input_shape
  |> normalisation ~decay:0.9
  |> conv2d [|3;3;3;32|] [|1;1|] ~act_typ:Activation.Relu
  |> conv2d [|3;3;32;32|] [|1;1|] ~act_typ:Activation.Relu ~padding:VALID
  |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
  |> dropout 0.1
  |> conv2d [|3;3;32;64|] [|1;1|] ~act_typ:Activation.Relu
  |> conv2d [|3;3;64;64|] [|1;1|] ~act_typ:Activation.Relu ~padding:VALID
  |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
  |> dropout 0.1
  |> fully_connected 512 ~act_typ:Activation.Relu
  |> linear 10 ~act_typ:Activation.(Softmax 1)
  |> get_network


let test network = 
  let x, _, y = Dataset.load_cifar_train_data 1 in
  let x, _ = Dataset.draw_samples_cifar x y 1 in 
  Owl_log.info "Testing network:... ";
  Graph.model network x |> Dense.Ndarray.S.print 


let train network =
  let x, _, y = Dataset.load_cifar_train_data 1 in
  let x, y = Dataset.draw_samples_cifar x y 200 in 

  let chkpt _state = () in 
  let params = Params.config
    ~batch:(Batch.Sample 100) 
    (* ~learning_rate:(Learning_Rate.Adagrad 0.01) *)
    ~learning_rate:(Learning_Rate.Adam (0.1, 0.9, 0.999))
    ~checkpoint:(Checkpoint.Custom chkpt) ~stopping:(Stopping.Const 1e-6) 10.
  in

  Owl_log.info "before training...";
  Dense.Ndarray.S.print (unpack_arr (Graph.mkpar network).(2).(0));

  Graph.train ~params ~init_model:false network x y |> ignore;

  Owl_log.info "after training...";
  Dense.Ndarray.S.print (unpack_arr (Graph.mkpar network).(2).(0))

(* 
let _ = 
  let network = make_network [|32;32;3|] in
  Graph.print network;
  Graph.init network;
  test network;
  train network;
  test network
*)

let network = make_network [|32;32;3|] 
let _ = Graph.print network
let _ = Graph.init network
let x, _, y = Dataset.load_cifar_train_data 1
(* let x, y = Dataset.draw_samples_cifar x y 200 *)

(* let _ = Dense.Ndarray.S.print y *)

let chkpt _state = ()
let params = Params.config
    ~batch:(Batch.Sample 100) 
    ~learning_rate:(Learning_Rate.Adagrad 0.001)
    (* ~learning_rate:(Learning_Rate.Adam (0.1, 0.9, 0.999)) *)
    ~checkpoint:(Checkpoint.Custom chkpt) ~stopping:(Stopping.Const 1e-6) 0.01

(* let _ = Owl_log.info "before training..."
let param = Graph.mkpar network;;
let _ = param.(16).(0) |> unpack_arr |> Dense.Ndarray.S.print *)

let _ = Graph.train ~params ~init_model:false network x y |> ignore

(* let _ = Owl_log.info "after training..."
let param = Graph.mkpar network;;
let _ = param.(16).(0) |> unpack_arr |> Dense.Ndarray.S.print *)

let _ = test network