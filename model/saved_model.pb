??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??
z
hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namehidden1/kernel
s
"hidden1/kernel/Read/ReadVariableOpReadVariableOphidden1/kernel* 
_output_shapes
:
??*
dtype0
q
hidden1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namehidden1/bias
j
 hidden1/bias/Read/ReadVariableOpReadVariableOphidden1/bias*
_output_shapes	
:?*
dtype0
y
hidden2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namehidden2/kernel
r
"hidden2/kernel/Read/ReadVariableOpReadVariableOphidden2/kernel*
_output_shapes
:	?@*
dtype0
p
hidden2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namehidden2/bias
i
 hidden2/bias/Read/ReadVariableOpReadVariableOphidden2/bias*
_output_shapes
:@*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@G*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:@G*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:G*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/hidden1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/hidden1/kernel/m
?
)Adam/hidden1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden1/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/hidden1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/hidden1/bias/m
x
'Adam/hidden1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/hidden2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameAdam/hidden2/kernel/m
?
)Adam/hidden2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden2/kernel/m*
_output_shapes
:	?@*
dtype0
~
Adam/hidden2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/hidden2/bias/m
w
'Adam/hidden2/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@G*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:@G*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:G*
dtype0
?
Adam/hidden1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/hidden1/kernel/v
?
)Adam/hidden1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden1/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/hidden1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/hidden1/bias/v
x
'Adam/hidden1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/hidden2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameAdam/hidden2/kernel/v
?
)Adam/hidden2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden2/kernel/v*
_output_shapes
:	?@*
dtype0
~
Adam/hidden2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/hidden2/bias/v
w
'Adam/hidden2/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@G*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:@G*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:G*
dtype0

NoOpNoOp
?'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?&
value?&B?& B?&
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
?
!iter

"beta_1

#beta_2
	$decay
%learning_ratemJmKmLmMmNmOvPvQvRvSvTvU
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
?
	variables
&metrics
trainable_variables
'layer_regularization_losses

(layers
)layer_metrics
regularization_losses
*non_trainable_variables
 
 
 
 
?
	variables
+metrics
trainable_variables

,layers
-layer_metrics
.non_trainable_variables
regularization_losses
/layer_regularization_losses
ZX
VARIABLE_VALUEhidden1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhidden1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
0metrics
trainable_variables

1layers
2layer_metrics
3non_trainable_variables
regularization_losses
4layer_regularization_losses
ZX
VARIABLE_VALUEhidden2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhidden2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
5metrics
trainable_variables

6layers
7layer_metrics
8non_trainable_variables
regularization_losses
9layer_regularization_losses
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
:metrics
trainable_variables

;layers
<layer_metrics
=non_trainable_variables
regularization_losses
>layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Atotal
	Bcount
C	variables
D	keras_api
D
	Etotal
	Fcount
G
_fn_kwargs
H	variables
I	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

C	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

H	variables
}{
VARIABLE_VALUEAdam/hidden1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/hidden1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/hidden2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/hidden2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/hidden1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/hidden1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/hidden2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/hidden2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_inputPlaceholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_inputhidden1/kernelhidden1/biashidden2/kernelhidden2/biasoutput/kerneloutput/bias*
Tin
	2*
Tout
2*'
_output_shapes
:?????????G*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*.
f)R'
%__inference_signature_wrapper_1119251
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"hidden1/kernel/Read/ReadVariableOp hidden1/bias/Read/ReadVariableOp"hidden2/kernel/Read/ReadVariableOp hidden2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/hidden1/kernel/m/Read/ReadVariableOp'Adam/hidden1/bias/m/Read/ReadVariableOp)Adam/hidden2/kernel/m/Read/ReadVariableOp'Adam/hidden2/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp)Adam/hidden1/kernel/v/Read/ReadVariableOp'Adam/hidden1/bias/v/Read/ReadVariableOp)Adam/hidden2/kernel/v/Read/ReadVariableOp'Adam/hidden2/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_save_1119518
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden1/kernelhidden1/biashidden2/kernelhidden2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/hidden1/kernel/mAdam/hidden1/bias/mAdam/hidden2/kernel/mAdam/hidden2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/hidden1/kernel/vAdam/hidden1/bias/vAdam/hidden2/kernel/vAdam/hidden2/bias/vAdam/output/kernel/vAdam/output/bias/v*'
Tin 
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference__traced_restore_1119611??
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_1119172

inputs
hidden1_1119156
hidden1_1119158
hidden2_1119161
hidden2_1119163
output_1119166
output_1119168
identity??hidden1/StatefulPartitionedCall?hidden2/StatefulPartitionedCall?output/StatefulPartitionedCall?
input/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_input_layer_call_and_return_conditional_losses_11190392
input/PartitionedCall?
hidden1/StatefulPartitionedCallStatefulPartitionedCallinput/PartitionedCall:output:0hidden1_1119156hidden1_1119158*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_hidden1_layer_call_and_return_conditional_losses_11190582!
hidden1/StatefulPartitionedCall?
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0hidden2_1119161hidden2_1119163*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_hidden2_layer_call_and_return_conditional_losses_11190852!
hidden2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0output_1119166output_1119168*
Tin
2*
Tout
2*'
_output_shapes
:?????????G*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_11191122 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_output_layer_call_and_return_conditional_losses_1119401

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@G*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:G*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????G2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_1119278

inputs*
&hidden1_matmul_readvariableop_resource+
'hidden1_biasadd_readvariableop_resource*
&hidden2_matmul_readvariableop_resource+
'hidden2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity?k
input/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
input/Constz
input/ReshapeReshapeinputsinput/Const:output:0*
T0*(
_output_shapes
:??????????2
input/Reshape?
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
hidden1/MatMul/ReadVariableOp?
hidden1/MatMulMatMulinput/Reshape:output:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden1/MatMul?
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
hidden1/BiasAdd/ReadVariableOp?
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden1/BiasAddq
hidden1/ReluReluhidden1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden1/Relu?
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
hidden2/MatMul/ReadVariableOp?
hidden2/MatMulMatMulhidden1/Relu:activations:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden2/MatMul?
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
hidden2/BiasAdd/ReadVariableOp?
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden2/BiasAddp
hidden2/ReluReluhidden2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
hidden2/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@G*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulhidden2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:G*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????G2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????:::::::T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_1119209

inputs
hidden1_1119193
hidden1_1119195
hidden2_1119198
hidden2_1119200
output_1119203
output_1119205
identity??hidden1/StatefulPartitionedCall?hidden2/StatefulPartitionedCall?output/StatefulPartitionedCall?
input/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_input_layer_call_and_return_conditional_losses_11190392
input/PartitionedCall?
hidden1/StatefulPartitionedCallStatefulPartitionedCallinput/PartitionedCall:output:0hidden1_1119193hidden1_1119195*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_hidden1_layer_call_and_return_conditional_losses_11190582!
hidden1/StatefulPartitionedCall?
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0hidden2_1119198hidden2_1119200*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_hidden2_layer_call_and_return_conditional_losses_11190852!
hidden2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0output_1119203output_1119205*
Tin
2*
Tout
2*'
_output_shapes
:?????????G*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_11191122 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_1119149
input_input
hidden1_1119133
hidden1_1119135
hidden2_1119138
hidden2_1119140
output_1119143
output_1119145
identity??hidden1/StatefulPartitionedCall?hidden2/StatefulPartitionedCall?output/StatefulPartitionedCall?
input/PartitionedCallPartitionedCallinput_input*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_input_layer_call_and_return_conditional_losses_11190392
input/PartitionedCall?
hidden1/StatefulPartitionedCallStatefulPartitionedCallinput/PartitionedCall:output:0hidden1_1119133hidden1_1119135*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_hidden1_layer_call_and_return_conditional_losses_11190582!
hidden1/StatefulPartitionedCall?
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0hidden2_1119138hidden2_1119140*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_hidden2_layer_call_and_return_conditional_losses_11190852!
hidden2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0output_1119143output_1119145*
Tin
2*
Tout
2*'
_output_shapes
:?????????G*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_11191122 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
,
_output_shapes
:??????????
%
_user_specified_nameinput_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
}
(__inference_output_layer_call_fn_1119410

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????G*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_11191122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?w
?
#__inference__traced_restore_1119611
file_prefix#
assignvariableop_hidden1_kernel#
assignvariableop_1_hidden1_bias%
!assignvariableop_2_hidden2_kernel#
assignvariableop_3_hidden2_bias$
 assignvariableop_4_output_kernel"
assignvariableop_5_output_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_1-
)assignvariableop_15_adam_hidden1_kernel_m+
'assignvariableop_16_adam_hidden1_bias_m-
)assignvariableop_17_adam_hidden2_kernel_m+
'assignvariableop_18_adam_hidden2_bias_m,
(assignvariableop_19_adam_output_kernel_m*
&assignvariableop_20_adam_output_bias_m-
)assignvariableop_21_adam_hidden1_kernel_v+
'assignvariableop_22_adam_hidden1_bias_v-
)assignvariableop_23_adam_hidden2_kernel_v+
'assignvariableop_24_adam_hidden2_bias_v,
(assignvariableop_25_adam_output_kernel_v*
&assignvariableop_26_adam_output_bias_v
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_hidden1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_hidden1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_hidden2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_hidden2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_hidden1_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_hidden1_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_hidden2_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_hidden2_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_output_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_output_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_hidden1_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_hidden1_bias_vIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_hidden2_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_hidden2_bias_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_output_kernel_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_output_bias_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27?
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*?
_input_shapesp
n: :::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
,__inference_sequential_layer_call_fn_1119187
input_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????G*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_11191722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:??????????
%
_user_specified_nameinput_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
^
B__inference_input_layer_call_and_return_conditional_losses_1119039

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1119251
input_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????G*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__wrapped_model_11190292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:??????????
%
_user_specified_nameinput_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_hidden2_layer_call_and_return_conditional_losses_1119085

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_layer_call_fn_1119339

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????G*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_11192092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_output_layer_call_and_return_conditional_losses_1119112

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@G*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:G*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????G2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
^
B__inference_input_layer_call_and_return_conditional_losses_1119345

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_layer_call_fn_1119224
input_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????G*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_11192092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:??????????
%
_user_specified_nameinput_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_hidden1_layer_call_and_return_conditional_losses_1119058

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_1119305

inputs*
&hidden1_matmul_readvariableop_resource+
'hidden1_biasadd_readvariableop_resource*
&hidden2_matmul_readvariableop_resource+
'hidden2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity?k
input/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
input/Constz
input/ReshapeReshapeinputsinput/Const:output:0*
T0*(
_output_shapes
:??????????2
input/Reshape?
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
hidden1/MatMul/ReadVariableOp?
hidden1/MatMulMatMulinput/Reshape:output:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden1/MatMul?
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
hidden1/BiasAdd/ReadVariableOp?
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden1/BiasAddq
hidden1/ReluReluhidden1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden1/Relu?
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
hidden2/MatMul/ReadVariableOp?
hidden2/MatMulMatMulhidden1/Relu:activations:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden2/MatMul?
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
hidden2/BiasAdd/ReadVariableOp?
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
hidden2/BiasAddp
hidden2/ReluReluhidden2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
hidden2/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@G*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulhidden2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:G*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????G2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????:::::::T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
C
'__inference_input_layer_call_fn_1119350

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_input_layer_call_and_return_conditional_losses_11190392
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_hidden1_layer_call_fn_1119370

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_hidden1_layer_call_and_return_conditional_losses_11190582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_layer_call_fn_1119322

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????G*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_11191722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_hidden1_layer_call_and_return_conditional_losses_1119361

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
"__inference__wrapped_model_1119029
input_input5
1sequential_hidden1_matmul_readvariableop_resource6
2sequential_hidden1_biasadd_readvariableop_resource5
1sequential_hidden2_matmul_readvariableop_resource6
2sequential_hidden2_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??
sequential/input/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential/input/Const?
sequential/input/ReshapeReshapeinput_inputsequential/input/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/input/Reshape?
(sequential/hidden1/MatMul/ReadVariableOpReadVariableOp1sequential_hidden1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/hidden1/MatMul/ReadVariableOp?
sequential/hidden1/MatMulMatMul!sequential/input/Reshape:output:00sequential/hidden1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/hidden1/MatMul?
)sequential/hidden1/BiasAdd/ReadVariableOpReadVariableOp2sequential_hidden1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/hidden1/BiasAdd/ReadVariableOp?
sequential/hidden1/BiasAddBiasAdd#sequential/hidden1/MatMul:product:01sequential/hidden1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/hidden1/BiasAdd?
sequential/hidden1/ReluRelu#sequential/hidden1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/hidden1/Relu?
(sequential/hidden2/MatMul/ReadVariableOpReadVariableOp1sequential_hidden2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02*
(sequential/hidden2/MatMul/ReadVariableOp?
sequential/hidden2/MatMulMatMul%sequential/hidden1/Relu:activations:00sequential/hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/hidden2/MatMul?
)sequential/hidden2/BiasAdd/ReadVariableOpReadVariableOp2sequential_hidden2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential/hidden2/BiasAdd/ReadVariableOp?
sequential/hidden2/BiasAddBiasAdd#sequential/hidden2/MatMul:product:01sequential/hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/hidden2/BiasAdd?
sequential/hidden2/ReluRelu#sequential/hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential/hidden2/Relu?
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:@G*
dtype02)
'sequential/output/MatMul/ReadVariableOp?
sequential/output/MatMulMatMul%sequential/hidden2/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G2
sequential/output/MatMul?
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:G*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp?
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G2
sequential/output/BiasAdd?
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????G2
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????:::::::Y U
,
_output_shapes
:??????????
%
_user_specified_nameinput_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_layer_call_and_return_conditional_losses_1119129
input_input
hidden1_1119069
hidden1_1119071
hidden2_1119096
hidden2_1119098
output_1119123
output_1119125
identity??hidden1/StatefulPartitionedCall?hidden2/StatefulPartitionedCall?output/StatefulPartitionedCall?
input/PartitionedCallPartitionedCallinput_input*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_input_layer_call_and_return_conditional_losses_11190392
input/PartitionedCall?
hidden1/StatefulPartitionedCallStatefulPartitionedCallinput/PartitionedCall:output:0hidden1_1119069hidden1_1119071*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_hidden1_layer_call_and_return_conditional_losses_11190582!
hidden1/StatefulPartitionedCall?
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0hidden2_1119096hidden2_1119098*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_hidden2_layer_call_and_return_conditional_losses_11190852!
hidden2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0output_1119123output_1119125*
Tin
2*
Tout
2*'
_output_shapes
:?????????G*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_11191122 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:??????????::::::2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:Y U
,
_output_shapes
:??????????
%
_user_specified_nameinput_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
~
)__inference_hidden2_layer_call_fn_1119390

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_hidden2_layer_call_and_return_conditional_losses_11190852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_hidden2_layer_call_and_return_conditional_losses_1119381

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?D
?
 __inference__traced_save_1119518
file_prefix-
)savev2_hidden1_kernel_read_readvariableop+
'savev2_hidden1_bias_read_readvariableop-
)savev2_hidden2_kernel_read_readvariableop+
'savev2_hidden2_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_hidden1_kernel_m_read_readvariableop2
.savev2_adam_hidden1_bias_m_read_readvariableop4
0savev2_adam_hidden2_kernel_m_read_readvariableop2
.savev2_adam_hidden2_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop4
0savev2_adam_hidden1_kernel_v_read_readvariableop2
.savev2_adam_hidden1_bias_v_read_readvariableop4
0savev2_adam_hidden2_kernel_v_read_readvariableop2
.savev2_adam_hidden2_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_66273d8470a0409eb3f4d97108550877/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_hidden1_kernel_read_readvariableop'savev2_hidden1_bias_read_readvariableop)savev2_hidden2_kernel_read_readvariableop'savev2_hidden2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_hidden1_kernel_m_read_readvariableop.savev2_adam_hidden1_bias_m_read_readvariableop0savev2_adam_hidden2_kernel_m_read_readvariableop.savev2_adam_hidden2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop0savev2_adam_hidden1_kernel_v_read_readvariableop.savev2_adam_hidden1_bias_v_read_readvariableop0savev2_adam_hidden2_kernel_v_read_readvariableop.savev2_adam_hidden2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *)
dtypes
2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:	?@:@:@G:G: : : : : : : : : :
??:?:	?@:@:@G:G:
??:?:	?@:@:@G:G: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@G: 

_output_shapes
:G:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@G: 

_output_shapes
:G:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@G: 

_output_shapes
:G:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
H
input_input9
serving_default_input_input:0??????????:
output0
StatefulPartitionedCall:0?????????Gtensorflow/serving/predict:Ô
?$
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
V_default_save_signature
W__call__
*X&call_and_return_all_conditional_losses"?!
_tf_keras_sequential?!{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Flatten", "config": {"name": "input", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 512]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 71, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 512]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 512]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Flatten", "config": {"name": "input", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 512]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 71, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 512]}}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": [{"class_name": "SparseCategoricalAccuracy", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
trainable_variables
regularization_losses
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "input", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 512]}, "stateful": false, "config": {"name": "input", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 512]}, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
___call__
*`&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 71, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
!iter

"beta_1

#beta_2
	$decay
%learning_ratemJmKmLmMmNmOvPvQvRvSvTvU"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
&metrics
trainable_variables
'layer_regularization_losses

(layers
)layer_metrics
regularization_losses
*non_trainable_variables
W__call__
V_default_save_signature
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
+metrics
trainable_variables

,layers
-layer_metrics
.non_trainable_variables
regularization_losses
/layer_regularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
": 
??2hidden1/kernel
:?2hidden1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
0metrics
trainable_variables

1layers
2layer_metrics
3non_trainable_variables
regularization_losses
4layer_regularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
!:	?@2hidden2/kernel
:@2hidden2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
5metrics
trainable_variables

6layers
7layer_metrics
8non_trainable_variables
regularization_losses
9layer_regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
:@G2output/kernel
:G2output/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
:metrics
trainable_variables

;layers
<layer_metrics
=non_trainable_variables
regularization_losses
>layer_regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Atotal
	Bcount
C	variables
D	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	Etotal
	Fcount
G
_fn_kwargs
H	variables
I	keras_api"?
_tf_keras_metric?{"class_name": "SparseCategoricalAccuracy", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
.
A0
B1"
trackable_list_wrapper
-
C	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
-
H	variables"
_generic_user_object
':%
??2Adam/hidden1/kernel/m
 :?2Adam/hidden1/bias/m
&:$	?@2Adam/hidden2/kernel/m
:@2Adam/hidden2/bias/m
$:"@G2Adam/output/kernel/m
:G2Adam/output/bias/m
':%
??2Adam/hidden1/kernel/v
 :?2Adam/hidden1/bias/v
&:$	?@2Adam/hidden2/kernel/v
:@2Adam/hidden2/bias/v
$:"@G2Adam/output/kernel/v
:G2Adam/output/bias/v
?2?
"__inference__wrapped_model_1119029?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_input??????????
?2?
,__inference_sequential_layer_call_fn_1119339
,__inference_sequential_layer_call_fn_1119224
,__inference_sequential_layer_call_fn_1119187
,__inference_sequential_layer_call_fn_1119322?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_layer_call_and_return_conditional_losses_1119149
G__inference_sequential_layer_call_and_return_conditional_losses_1119129
G__inference_sequential_layer_call_and_return_conditional_losses_1119278
G__inference_sequential_layer_call_and_return_conditional_losses_1119305?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_input_layer_call_fn_1119350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_input_layer_call_and_return_conditional_losses_1119345?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_hidden1_layer_call_fn_1119370?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_hidden1_layer_call_and_return_conditional_losses_1119361?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_hidden2_layer_call_fn_1119390?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_hidden2_layer_call_and_return_conditional_losses_1119381?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_output_layer_call_fn_1119410?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_output_layer_call_and_return_conditional_losses_1119401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
8B6
%__inference_signature_wrapper_1119251input_input?
"__inference__wrapped_model_1119029t9?6
/?,
*?'
input_input??????????
? "/?,
*
output ?
output?????????G?
D__inference_hidden1_layer_call_and_return_conditional_losses_1119361^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_hidden1_layer_call_fn_1119370Q0?-
&?#
!?
inputs??????????
? "????????????
D__inference_hidden2_layer_call_and_return_conditional_losses_1119381]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? }
)__inference_hidden2_layer_call_fn_1119390P0?-
&?#
!?
inputs??????????
? "??????????@?
B__inference_input_layer_call_and_return_conditional_losses_1119345^4?1
*?'
%?"
inputs??????????
? "&?#
?
0??????????
? |
'__inference_input_layer_call_fn_1119350Q4?1
*?'
%?"
inputs??????????
? "????????????
C__inference_output_layer_call_and_return_conditional_losses_1119401\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????G
? {
(__inference_output_layer_call_fn_1119410O/?,
%?"
 ?
inputs?????????@
? "??????????G?
G__inference_sequential_layer_call_and_return_conditional_losses_1119129rA?>
7?4
*?'
input_input??????????
p

 
? "%?"
?
0?????????G
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_1119149rA?>
7?4
*?'
input_input??????????
p 

 
? "%?"
?
0?????????G
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_1119278m<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????G
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_1119305m<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????G
? ?
,__inference_sequential_layer_call_fn_1119187eA?>
7?4
*?'
input_input??????????
p

 
? "??????????G?
,__inference_sequential_layer_call_fn_1119224eA?>
7?4
*?'
input_input??????????
p 

 
? "??????????G?
,__inference_sequential_layer_call_fn_1119322`<?9
2?/
%?"
inputs??????????
p

 
? "??????????G?
,__inference_sequential_layer_call_fn_1119339`<?9
2?/
%?"
inputs??????????
p 

 
? "??????????G?
%__inference_signature_wrapper_1119251?H?E
? 
>?;
9
input_input*?'
input_input??????????"/?,
*
output ?
output?????????G