open Owl
open Neural.S
open Neural.S.Graph
open Owl_algodiff.S
module N = Dense.Ndarray.S

let shp = [|100;10|]
let carr = N.ones shp

let loss x = 
	let foo = Maths.(mul (Arr carr) x |> sum' |> neg) in
	let bar = Maths.(div (F 100.) foo) in
	Maths.add (F 0.) bar


let inp  = DR ((Arr (N.ones shp)), ref (Arr (N.zeros shp)), Noop, ref 0, 0)
let outp = loss inp

let _ = reverse_prop (F 1.) outp