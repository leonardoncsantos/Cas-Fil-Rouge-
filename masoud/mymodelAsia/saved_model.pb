��	
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
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
�
Adam/v/dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_43/bias
y
(Adam/v/dense_43/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_43/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_43/bias
y
(Adam/m/dense_43/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_43/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:6*'
shared_nameAdam/v/dense_43/kernel
�
*Adam/v/dense_43/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_43/kernel*
_output_shapes

:6*
dtype0
�
Adam/m/dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:6*'
shared_nameAdam/m/dense_43/kernel
�
*Adam/m/dense_43/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_43/kernel*
_output_shapes

:6*
dtype0
�
Adam/v/dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:6*%
shared_nameAdam/v/dense_41/bias
y
(Adam/v/dense_41/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_41/bias*
_output_shapes
:6*
dtype0
�
Adam/m/dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:6*%
shared_nameAdam/m/dense_41/bias
y
(Adam/m/dense_41/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_41/bias*
_output_shapes
:6*
dtype0
�
Adam/v/dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$6*'
shared_nameAdam/v/dense_41/kernel
�
*Adam/v/dense_41/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_41/kernel*
_output_shapes

:$6*
dtype0
�
Adam/m/dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$6*'
shared_nameAdam/m/dense_41/kernel
�
*Adam/m/dense_41/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_41/kernel*
_output_shapes

:$6*
dtype0
�
"Adam/v/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*3
shared_name$"Adam/v/batch_normalization_10/beta
�
6Adam/v/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_10/beta*
_output_shapes
:$*
dtype0
�
"Adam/m/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*3
shared_name$"Adam/m/batch_normalization_10/beta
�
6Adam/m/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_10/beta*
_output_shapes
:$*
dtype0
�
#Adam/v/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*4
shared_name%#Adam/v/batch_normalization_10/gamma
�
7Adam/v/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_10/gamma*
_output_shapes
:$*
dtype0
�
#Adam/m/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*4
shared_name%#Adam/m/batch_normalization_10/gamma
�
7Adam/m/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_10/gamma*
_output_shapes
:$*
dtype0
�
Adam/v/dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/v/dense_40/bias
y
(Adam/v/dense_40/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_40/bias*
_output_shapes
:$*
dtype0
�
Adam/m/dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*%
shared_nameAdam/m/dense_40/bias
y
(Adam/m/dense_40/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_40/bias*
_output_shapes
:$*
dtype0
�
Adam/v/dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*'
shared_nameAdam/v/dense_40/kernel
�
*Adam/v/dense_40/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_40/kernel*
_output_shapes

:	$*
dtype0
�
Adam/m/dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$*'
shared_nameAdam/m/dense_40/kernel
�
*Adam/m/dense_40/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_40/kernel*
_output_shapes

:	$*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_43/bias
k
!dense_43/bias/Read/ReadVariableOpReadVariableOpdense_43/bias*
_output_shapes
:*
dtype0
z
dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:6* 
shared_namedense_43/kernel
s
#dense_43/kernel/Read/ReadVariableOpReadVariableOpdense_43/kernel*
_output_shapes

:6*
dtype0
r
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:6*
shared_namedense_41/bias
k
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
_output_shapes
:6*
dtype0
z
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$6* 
shared_namedense_41/kernel
s
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel*
_output_shapes

:$6*
dtype0
�
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*7
shared_name(&batch_normalization_10/moving_variance
�
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:$*
dtype0
�
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*3
shared_name$"batch_normalization_10/moving_mean
�
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:$*
dtype0
�
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*,
shared_namebatch_normalization_10/beta
�
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:$*
dtype0
�
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*-
shared_namebatch_normalization_10/gamma
�
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:$*
dtype0
r
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_namedense_40/bias
k
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes
:$*
dtype0
z
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	$* 
shared_namedense_40/kernel
s
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes

:	$*
dtype0
{
serving_default_input_11Placeholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11dense_40/kerneldense_40/bias&batch_normalization_10/moving_variancebatch_normalization_10/gamma"batch_normalization_10/moving_meanbatch_normalization_10/betadense_41/kerneldense_41/biasdense_43/kerneldense_43/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2223275

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�:
value�:B�: B�:
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	gamma
 beta
!moving_mean
"moving_variance*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
J
0
1
2
 3
!4
"5
/6
07
78
89*
<
0
1
2
 3
/4
05
76
87*
* 
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
6
Btrace_0
Ctrace_1
Dtrace_2
Etrace_3* 
* 
�
F
_variables
G_iterations
H_learning_rate
I_index_dict
J
_momentums
K_velocities
L_update_step_xla*

Mserving_default* 

0
1*

0
1*
* 
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Strace_0* 

Ttrace_0* 
_Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_40/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
0
 1
!2
"3*

0
 1*
* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ztrace_0
[trace_1* 

\trace_0
]trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

ctrace_0* 

dtrace_0* 

/0
01*

/0
01*
* 
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

jtrace_0* 

ktrace_0* 
_Y
VARIABLE_VALUEdense_41/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_41/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

qtrace_0* 

rtrace_0* 
_Y
VARIABLE_VALUEdense_43/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_43/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

!0
"1*
.
0
1
2
3
4
5*

s0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
G0
t1
u2
v3
w4
x5
y6
z7
{8
|9
}10
~11
12
�13
�14
�15
�16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
>
t0
v1
x2
z3
|4
~5
�6
�7*
>
u0
w1
y2
{3
}4
5
�6
�7*
* 
* 
* 
* 
* 
* 
* 
* 
* 

!0
"1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
a[
VARIABLE_VALUEAdam/m/dense_40/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_40/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_40/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_40/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/m/batch_normalization_10/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/v/batch_normalization_10/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/batch_normalization_10/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/batch_normalization_10/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_41/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_41/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_41/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_41/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_43/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_43/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_43/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_43/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_40/kerneldense_40/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variancedense_41/kerneldense_41/biasdense_43/kerneldense_43/bias	iterationlearning_rateAdam/m/dense_40/kernelAdam/v/dense_40/kernelAdam/m/dense_40/biasAdam/v/dense_40/bias#Adam/m/batch_normalization_10/gamma#Adam/v/batch_normalization_10/gamma"Adam/m/batch_normalization_10/beta"Adam/v/batch_normalization_10/betaAdam/m/dense_41/kernelAdam/v/dense_41/kernelAdam/m/dense_41/biasAdam/v/dense_41/biasAdam/m/dense_43/kernelAdam/v/dense_43/kernelAdam/m/dense_43/biasAdam/v/dense_43/biastotalcountConst*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_2223767
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_40/kerneldense_40/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variancedense_41/kerneldense_41/biasdense_43/kerneldense_43/bias	iterationlearning_rateAdam/m/dense_40/kernelAdam/v/dense_40/kernelAdam/m/dense_40/biasAdam/v/dense_40/bias#Adam/m/batch_normalization_10/gamma#Adam/v/batch_normalization_10/gamma"Adam/m/batch_normalization_10/beta"Adam/v/batch_normalization_10/betaAdam/m/dense_41/kernelAdam/v/dense_41/kernelAdam/m/dense_41/biasAdam/v/dense_41/biasAdam/m/dense_43/kernelAdam/v/dense_43/kernelAdam/m/dense_43/biasAdam/v/dense_43/biastotalcount**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_2223867��
�
a
E__inference_re_lu_10_layer_call_and_return_conditional_losses_2223526

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������$Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2222906

inputs5
'assignmovingavg_readvariableop_resource:$7
)assignmovingavg_1_readvariableop_resource:$3
%batchnorm_mul_readvariableop_resource:$/
!batchnorm_readvariableop_resource:$
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:$�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������$l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:$*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:$x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:$�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:$*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:$~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:$�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������$h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:$v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������$b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������$�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������$: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�

�
*__inference_model_10_layer_call_fn_2223160
input_11
unknown:	$
	unknown_0:$
	unknown_1:$
	unknown_2:$
	unknown_3:$
	unknown_4:$
	unknown_5:$6
	unknown_6:6
	unknown_7:6
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_2223137o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
input_11
�

�
*__inference_model_10_layer_call_fn_2223325

inputs
unknown:	$
	unknown_0:$
	unknown_1:$
	unknown_2:$
	unknown_3:$
	unknown_4:$
	unknown_5:$6
	unknown_6:6
	unknown_7:6
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_2223137o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
*__inference_dense_40_layer_call_fn_2223426

inputs
unknown:	$
	unknown_0:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_2222967o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�3
�
E__inference_model_10_layer_call_and_return_conditional_losses_2223417

inputs9
'dense_40_matmul_readvariableop_resource:	$6
(dense_40_biasadd_readvariableop_resource:$F
8batch_normalization_10_batchnorm_readvariableop_resource:$J
<batch_normalization_10_batchnorm_mul_readvariableop_resource:$H
:batch_normalization_10_batchnorm_readvariableop_1_resource:$H
:batch_normalization_10_batchnorm_readvariableop_2_resource:$9
'dense_41_matmul_readvariableop_resource:$66
(dense_41_biasadd_readvariableop_resource:69
'dense_43_matmul_readvariableop_resource:66
(dense_43_biasadd_readvariableop_resource:
identity��/batch_normalization_10/batchnorm/ReadVariableOp�1batch_normalization_10/batchnorm/ReadVariableOp_1�1batch_normalization_10/batchnorm/ReadVariableOp_2�3batch_normalization_10/batchnorm/mul/ReadVariableOp�dense_40/BiasAdd/ReadVariableOp�dense_40/MatMul/ReadVariableOp�dense_41/BiasAdd/ReadVariableOp�dense_41/MatMul/ReadVariableOp�dense_43/BiasAdd/ReadVariableOp�dense_43/MatMul/ReadVariableOp�
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype0{
dense_40/MatMulMatMulinputs&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$�
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0�
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$�
/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0k
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_10/batchnorm/addAddV27batch_normalization_10/batchnorm/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:$~
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:$�
3batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:0;batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$�
&batch_normalization_10/batchnorm/mul_1Muldense_40/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������$�
1batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype0�
&batch_normalization_10/batchnorm/mul_2Mul9batch_normalization_10/batchnorm/ReadVariableOp_1:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:$�
1batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype0�
$batch_normalization_10/batchnorm/subSub9batch_normalization_10/batchnorm/ReadVariableOp_2:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:$�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������$s
re_lu_10/ReluRelu*batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������$�
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:$6*
dtype0�
dense_41/MatMulMatMulre_lu_10/Relu:activations:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������6�
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype0�
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������6�
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes

:6*
dtype0�
dense_43/MatMulMatMuldense_41/BiasAdd:output:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_43/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^batch_normalization_10/batchnorm/ReadVariableOp2^batch_normalization_10/batchnorm/ReadVariableOp_12^batch_normalization_10/batchnorm/ReadVariableOp_24^batch_normalization_10/batchnorm/mul/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 2b
/batch_normalization_10/batchnorm/ReadVariableOp/batch_normalization_10/batchnorm/ReadVariableOp2f
1batch_normalization_10/batchnorm/ReadVariableOp_11batch_normalization_10/batchnorm/ReadVariableOp_12f
1batch_normalization_10/batchnorm/ReadVariableOp_21batch_normalization_10/batchnorm/ReadVariableOp_22j
3batch_normalization_10/batchnorm/mul/ReadVariableOp3batch_normalization_10/batchnorm/mul/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
�
E__inference_dense_43_layer_call_and_return_conditional_losses_2223015

inputs0
matmul_readvariableop_resource:6-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:6*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������6: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������6
 
_user_specified_nameinputs
�
�
E__inference_model_10_layer_call_and_return_conditional_losses_2223083

inputs"
dense_40_2223057:	$
dense_40_2223059:$,
batch_normalization_10_2223062:$,
batch_normalization_10_2223064:$,
batch_normalization_10_2223066:$,
batch_normalization_10_2223068:$"
dense_41_2223072:$6
dense_41_2223074:6"
dense_43_2223077:6
dense_43_2223079:
identity��.batch_normalization_10/StatefulPartitionedCall� dense_40/StatefulPartitionedCall� dense_41/StatefulPartitionedCall� dense_43/StatefulPartitionedCall�
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinputsdense_40_2223057dense_40_2223059*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_2222967�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0batch_normalization_10_2223062batch_normalization_10_2223064batch_normalization_10_2223066batch_normalization_10_2223068*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2222906�
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_2222987�
 dense_41/StatefulPartitionedCallStatefulPartitionedCall!re_lu_10/PartitionedCall:output:0dense_41_2223072dense_41_2223074*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_2222999�
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_43_2223077dense_43_2223079*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_2223015x
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�;
�

"__inference__wrapped_model_2222871
input_11B
0model_10_dense_40_matmul_readvariableop_resource:	$?
1model_10_dense_40_biasadd_readvariableop_resource:$O
Amodel_10_batch_normalization_10_batchnorm_readvariableop_resource:$S
Emodel_10_batch_normalization_10_batchnorm_mul_readvariableop_resource:$Q
Cmodel_10_batch_normalization_10_batchnorm_readvariableop_1_resource:$Q
Cmodel_10_batch_normalization_10_batchnorm_readvariableop_2_resource:$B
0model_10_dense_41_matmul_readvariableop_resource:$6?
1model_10_dense_41_biasadd_readvariableop_resource:6B
0model_10_dense_43_matmul_readvariableop_resource:6?
1model_10_dense_43_biasadd_readvariableop_resource:
identity��8model_10/batch_normalization_10/batchnorm/ReadVariableOp�:model_10/batch_normalization_10/batchnorm/ReadVariableOp_1�:model_10/batch_normalization_10/batchnorm/ReadVariableOp_2�<model_10/batch_normalization_10/batchnorm/mul/ReadVariableOp�(model_10/dense_40/BiasAdd/ReadVariableOp�'model_10/dense_40/MatMul/ReadVariableOp�(model_10/dense_41/BiasAdd/ReadVariableOp�'model_10/dense_41/MatMul/ReadVariableOp�(model_10/dense_43/BiasAdd/ReadVariableOp�'model_10/dense_43/MatMul/ReadVariableOp�
'model_10/dense_40/MatMul/ReadVariableOpReadVariableOp0model_10_dense_40_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype0�
model_10/dense_40/MatMulMatMulinput_11/model_10/dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$�
(model_10/dense_40/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_40_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0�
model_10/dense_40/BiasAddBiasAdd"model_10/dense_40/MatMul:product:00model_10/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$�
8model_10/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOpAmodel_10_batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0t
/model_10/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-model_10/batch_normalization_10/batchnorm/addAddV2@model_10/batch_normalization_10/batchnorm/ReadVariableOp:value:08model_10/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:$�
/model_10/batch_normalization_10/batchnorm/RsqrtRsqrt1model_10/batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:$�
<model_10/batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_10_batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0�
-model_10/batch_normalization_10/batchnorm/mulMul3model_10/batch_normalization_10/batchnorm/Rsqrt:y:0Dmodel_10/batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$�
/model_10/batch_normalization_10/batchnorm/mul_1Mul"model_10/dense_40/BiasAdd:output:01model_10/batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������$�
:model_10/batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_10_batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype0�
/model_10/batch_normalization_10/batchnorm/mul_2MulBmodel_10/batch_normalization_10/batchnorm/ReadVariableOp_1:value:01model_10/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:$�
:model_10/batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_10_batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype0�
-model_10/batch_normalization_10/batchnorm/subSubBmodel_10/batch_normalization_10/batchnorm/ReadVariableOp_2:value:03model_10/batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:$�
/model_10/batch_normalization_10/batchnorm/add_1AddV23model_10/batch_normalization_10/batchnorm/mul_1:z:01model_10/batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������$�
model_10/re_lu_10/ReluRelu3model_10/batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������$�
'model_10/dense_41/MatMul/ReadVariableOpReadVariableOp0model_10_dense_41_matmul_readvariableop_resource*
_output_shapes

:$6*
dtype0�
model_10/dense_41/MatMulMatMul$model_10/re_lu_10/Relu:activations:0/model_10/dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������6�
(model_10/dense_41/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_41_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype0�
model_10/dense_41/BiasAddBiasAdd"model_10/dense_41/MatMul:product:00model_10/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������6�
'model_10/dense_43/MatMul/ReadVariableOpReadVariableOp0model_10_dense_43_matmul_readvariableop_resource*
_output_shapes

:6*
dtype0�
model_10/dense_43/MatMulMatMul"model_10/dense_41/BiasAdd:output:0/model_10/dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(model_10/dense_43/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_10/dense_43/BiasAddBiasAdd"model_10/dense_43/MatMul:product:00model_10/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"model_10/dense_43/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^model_10/batch_normalization_10/batchnorm/ReadVariableOp;^model_10/batch_normalization_10/batchnorm/ReadVariableOp_1;^model_10/batch_normalization_10/batchnorm/ReadVariableOp_2=^model_10/batch_normalization_10/batchnorm/mul/ReadVariableOp)^model_10/dense_40/BiasAdd/ReadVariableOp(^model_10/dense_40/MatMul/ReadVariableOp)^model_10/dense_41/BiasAdd/ReadVariableOp(^model_10/dense_41/MatMul/ReadVariableOp)^model_10/dense_43/BiasAdd/ReadVariableOp(^model_10/dense_43/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 2t
8model_10/batch_normalization_10/batchnorm/ReadVariableOp8model_10/batch_normalization_10/batchnorm/ReadVariableOp2x
:model_10/batch_normalization_10/batchnorm/ReadVariableOp_1:model_10/batch_normalization_10/batchnorm/ReadVariableOp_12x
:model_10/batch_normalization_10/batchnorm/ReadVariableOp_2:model_10/batch_normalization_10/batchnorm/ReadVariableOp_22|
<model_10/batch_normalization_10/batchnorm/mul/ReadVariableOp<model_10/batch_normalization_10/batchnorm/mul/ReadVariableOp2T
(model_10/dense_40/BiasAdd/ReadVariableOp(model_10/dense_40/BiasAdd/ReadVariableOp2R
'model_10/dense_40/MatMul/ReadVariableOp'model_10/dense_40/MatMul/ReadVariableOp2T
(model_10/dense_41/BiasAdd/ReadVariableOp(model_10/dense_41/BiasAdd/ReadVariableOp2R
'model_10/dense_41/MatMul/ReadVariableOp'model_10/dense_41/MatMul/ReadVariableOp2T
(model_10/dense_43/BiasAdd/ReadVariableOp(model_10/dense_43/BiasAdd/ReadVariableOp2R
'model_10/dense_43/MatMul/ReadVariableOp'model_10/dense_43/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������	
"
_user_specified_name
input_11
�
�
*__inference_dense_41_layer_call_fn_2223535

inputs
unknown:$6
	unknown_0:6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_2222999o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������6`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������$: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
��
�
 __inference__traced_save_2223767
file_prefix8
&read_disablecopyonread_dense_40_kernel:	$4
&read_1_disablecopyonread_dense_40_bias:$C
5read_2_disablecopyonread_batch_normalization_10_gamma:$B
4read_3_disablecopyonread_batch_normalization_10_beta:$I
;read_4_disablecopyonread_batch_normalization_10_moving_mean:$M
?read_5_disablecopyonread_batch_normalization_10_moving_variance:$:
(read_6_disablecopyonread_dense_41_kernel:$64
&read_7_disablecopyonread_dense_41_bias:6:
(read_8_disablecopyonread_dense_43_kernel:64
&read_9_disablecopyonread_dense_43_bias:-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: B
0read_12_disablecopyonread_adam_m_dense_40_kernel:	$B
0read_13_disablecopyonread_adam_v_dense_40_kernel:	$<
.read_14_disablecopyonread_adam_m_dense_40_bias:$<
.read_15_disablecopyonread_adam_v_dense_40_bias:$K
=read_16_disablecopyonread_adam_m_batch_normalization_10_gamma:$K
=read_17_disablecopyonread_adam_v_batch_normalization_10_gamma:$J
<read_18_disablecopyonread_adam_m_batch_normalization_10_beta:$J
<read_19_disablecopyonread_adam_v_batch_normalization_10_beta:$B
0read_20_disablecopyonread_adam_m_dense_41_kernel:$6B
0read_21_disablecopyonread_adam_v_dense_41_kernel:$6<
.read_22_disablecopyonread_adam_m_dense_41_bias:6<
.read_23_disablecopyonread_adam_v_dense_41_bias:6B
0read_24_disablecopyonread_adam_m_dense_43_kernel:6B
0read_25_disablecopyonread_adam_v_dense_43_kernel:6<
.read_26_disablecopyonread_adam_m_dense_43_bias:<
.read_27_disablecopyonread_adam_v_dense_43_bias:)
read_28_disablecopyonread_total: )
read_29_disablecopyonread_count: 
savev2_const
identity_61��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_40_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_40_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:	$*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:	$a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:	$z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_40_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_40_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_2/DisableCopyOnReadDisableCopyOnRead5read_2_disablecopyonread_batch_normalization_10_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp5read_2_disablecopyonread_batch_normalization_10_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_3/DisableCopyOnReadDisableCopyOnRead4read_3_disablecopyonread_batch_normalization_10_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp4read_3_disablecopyonread_batch_normalization_10_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_4/DisableCopyOnReadDisableCopyOnRead;read_4_disablecopyonread_batch_normalization_10_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp;read_4_disablecopyonread_batch_normalization_10_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_5/DisableCopyOnReadDisableCopyOnRead?read_5_disablecopyonread_batch_normalization_10_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp?read_5_disablecopyonread_batch_normalization_10_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:$|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_dense_41_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_dense_41_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:$6*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:$6e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:$6z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_dense_41_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_dense_41_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:6*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:6a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:6|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_dense_43_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_dense_43_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:6*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:6e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:6z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_dense_43_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_dense_43_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_iteration^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnRead0read_12_disablecopyonread_adam_m_dense_40_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp0read_12_disablecopyonread_adam_m_dense_40_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:	$*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:	$e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:	$�
Read_13/DisableCopyOnReadDisableCopyOnRead0read_13_disablecopyonread_adam_v_dense_40_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp0read_13_disablecopyonread_adam_v_dense_40_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:	$*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:	$e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:	$�
Read_14/DisableCopyOnReadDisableCopyOnRead.read_14_disablecopyonread_adam_m_dense_40_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp.read_14_disablecopyonread_adam_m_dense_40_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_15/DisableCopyOnReadDisableCopyOnRead.read_15_disablecopyonread_adam_v_dense_40_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp.read_15_disablecopyonread_adam_v_dense_40_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_16/DisableCopyOnReadDisableCopyOnRead=read_16_disablecopyonread_adam_m_batch_normalization_10_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp=read_16_disablecopyonread_adam_m_batch_normalization_10_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_17/DisableCopyOnReadDisableCopyOnRead=read_17_disablecopyonread_adam_v_batch_normalization_10_gamma"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp=read_17_disablecopyonread_adam_v_batch_normalization_10_gamma^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_18/DisableCopyOnReadDisableCopyOnRead<read_18_disablecopyonread_adam_m_batch_normalization_10_beta"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp<read_18_disablecopyonread_adam_m_batch_normalization_10_beta^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_19/DisableCopyOnReadDisableCopyOnRead<read_19_disablecopyonread_adam_v_batch_normalization_10_beta"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp<read_19_disablecopyonread_adam_v_batch_normalization_10_beta^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_adam_m_dense_41_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_adam_m_dense_41_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:$6*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:$6e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:$6�
Read_21/DisableCopyOnReadDisableCopyOnRead0read_21_disablecopyonread_adam_v_dense_41_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp0read_21_disablecopyonread_adam_v_dense_41_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:$6*
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:$6e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

:$6�
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_adam_m_dense_41_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_adam_m_dense_41_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:6*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:6a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:6�
Read_23/DisableCopyOnReadDisableCopyOnRead.read_23_disablecopyonread_adam_v_dense_41_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp.read_23_disablecopyonread_adam_v_dense_41_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:6*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:6a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:6�
Read_24/DisableCopyOnReadDisableCopyOnRead0read_24_disablecopyonread_adam_m_dense_43_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp0read_24_disablecopyonread_adam_m_dense_43_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:6*
dtype0o
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:6e
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes

:6�
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_adam_v_dense_43_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_adam_v_dense_43_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:6*
dtype0o
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:6e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:6�
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_m_dense_43_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_m_dense_43_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_27/DisableCopyOnReadDisableCopyOnRead.read_27_disablecopyonread_adam_v_dense_43_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp.read_27_disablecopyonread_adam_v_dense_43_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_28/DisableCopyOnReadDisableCopyOnReadread_28_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOpread_28_disablecopyonread_total^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_count^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_60Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_61IdentityIdentity_60:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_61Identity_61:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 

�
#__inference__traced_restore_2223867
file_prefix2
 assignvariableop_dense_40_kernel:	$.
 assignvariableop_1_dense_40_bias:$=
/assignvariableop_2_batch_normalization_10_gamma:$<
.assignvariableop_3_batch_normalization_10_beta:$C
5assignvariableop_4_batch_normalization_10_moving_mean:$G
9assignvariableop_5_batch_normalization_10_moving_variance:$4
"assignvariableop_6_dense_41_kernel:$6.
 assignvariableop_7_dense_41_bias:64
"assignvariableop_8_dense_43_kernel:6.
 assignvariableop_9_dense_43_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: <
*assignvariableop_12_adam_m_dense_40_kernel:	$<
*assignvariableop_13_adam_v_dense_40_kernel:	$6
(assignvariableop_14_adam_m_dense_40_bias:$6
(assignvariableop_15_adam_v_dense_40_bias:$E
7assignvariableop_16_adam_m_batch_normalization_10_gamma:$E
7assignvariableop_17_adam_v_batch_normalization_10_gamma:$D
6assignvariableop_18_adam_m_batch_normalization_10_beta:$D
6assignvariableop_19_adam_v_batch_normalization_10_beta:$<
*assignvariableop_20_adam_m_dense_41_kernel:$6<
*assignvariableop_21_adam_v_dense_41_kernel:$66
(assignvariableop_22_adam_m_dense_41_bias:66
(assignvariableop_23_adam_v_dense_41_bias:6<
*assignvariableop_24_adam_m_dense_43_kernel:6<
*assignvariableop_25_adam_v_dense_43_kernel:66
(assignvariableop_26_adam_m_dense_43_bias:6
(assignvariableop_27_adam_v_dense_43_bias:#
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_40_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_40_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_10_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_10_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_10_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_10_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_41_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_41_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_43_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_43_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_m_dense_40_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_v_dense_40_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_m_dense_40_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_v_dense_40_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp7assignvariableop_16_adam_m_batch_normalization_10_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adam_v_batch_normalization_10_gammaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_m_batch_normalization_10_betaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_v_batch_normalization_10_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_m_dense_41_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_v_dense_41_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_m_dense_41_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_v_dense_41_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_m_dense_43_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_v_dense_43_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_m_dense_43_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_v_dense_43_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
F
*__inference_re_lu_10_layer_call_fn_2223521

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_2222987`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�

�
*__inference_model_10_layer_call_fn_2223106
input_11
unknown:	$
	unknown_0:$
	unknown_1:$
	unknown_2:$
	unknown_3:$
	unknown_4:$
	unknown_5:$6
	unknown_6:6
	unknown_7:6
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_2223083o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
input_11
�
�
E__inference_model_10_layer_call_and_return_conditional_losses_2223022
input_11"
dense_40_2222968:	$
dense_40_2222970:$,
batch_normalization_10_2222973:$,
batch_normalization_10_2222975:$,
batch_normalization_10_2222977:$,
batch_normalization_10_2222979:$"
dense_41_2223000:$6
dense_41_2223002:6"
dense_43_2223016:6
dense_43_2223018:
identity��.batch_normalization_10/StatefulPartitionedCall� dense_40/StatefulPartitionedCall� dense_41/StatefulPartitionedCall� dense_43/StatefulPartitionedCall�
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinput_11dense_40_2222968dense_40_2222970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_2222967�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0batch_normalization_10_2222973batch_normalization_10_2222975batch_normalization_10_2222977batch_normalization_10_2222979*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2222906�
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_2222987�
 dense_41/StatefulPartitionedCallStatefulPartitionedCall!re_lu_10/PartitionedCall:output:0dense_41_2223000dense_41_2223002*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_2222999�
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_43_2223016dense_43_2223018*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_2223015x
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
input_11
�
�
8__inference_batch_normalization_10_layer_call_fn_2223449

inputs
unknown:$
	unknown_0:$
	unknown_1:$
	unknown_2:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2222906o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������$: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
E__inference_dense_43_layer_call_and_return_conditional_losses_2223564

inputs0
matmul_readvariableop_resource:6-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:6*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������6: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������6
 
_user_specified_nameinputs
�
�
*__inference_dense_43_layer_call_fn_2223554

inputs
unknown:6
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_2223015o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������6: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������6
 
_user_specified_nameinputs
�	
�
E__inference_dense_40_layer_call_and_return_conditional_losses_2222967

inputs0
matmul_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2222926

inputs/
!batchnorm_readvariableop_resource:$3
%batchnorm_mul_readvariableop_resource:$1
#batchnorm_readvariableop_1_resource:$1
#batchnorm_readvariableop_2_resource:$
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������$z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:$z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������$b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������$�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������$: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_10_layer_call_fn_2223462

inputs
unknown:$
	unknown_0:$
	unknown_1:$
	unknown_2:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2222926o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������$: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
E__inference_model_10_layer_call_and_return_conditional_losses_2223051
input_11"
dense_40_2223025:	$
dense_40_2223027:$,
batch_normalization_10_2223030:$,
batch_normalization_10_2223032:$,
batch_normalization_10_2223034:$,
batch_normalization_10_2223036:$"
dense_41_2223040:$6
dense_41_2223042:6"
dense_43_2223045:6
dense_43_2223047:
identity��.batch_normalization_10/StatefulPartitionedCall� dense_40/StatefulPartitionedCall� dense_41/StatefulPartitionedCall� dense_43/StatefulPartitionedCall�
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinput_11dense_40_2223025dense_40_2223027*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_2222967�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0batch_normalization_10_2223030batch_normalization_10_2223032batch_normalization_10_2223034batch_normalization_10_2223036*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2222926�
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_2222987�
 dense_41/StatefulPartitionedCallStatefulPartitionedCall!re_lu_10/PartitionedCall:output:0dense_41_2223040dense_41_2223042*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_2222999�
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_43_2223045dense_43_2223047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_2223015x
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
input_11
�	
�
E__inference_dense_40_layer_call_and_return_conditional_losses_2223436

inputs0
matmul_readvariableop_resource:	$-
biasadd_readvariableop_resource:$
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
�
%__inference_signature_wrapper_2223275
input_11
unknown:	$
	unknown_0:$
	unknown_1:$
	unknown_2:$
	unknown_3:$
	unknown_4:$
	unknown_5:$6
	unknown_6:6
	unknown_7:6
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2222871o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
input_11
�
�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2223516

inputs/
!batchnorm_readvariableop_resource:$3
%batchnorm_mul_readvariableop_resource:$1
#batchnorm_readvariableop_1_resource:$1
#batchnorm_readvariableop_2_resource:$
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������$z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:$z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������$b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������$�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������$: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
E__inference_dense_41_layer_call_and_return_conditional_losses_2223545

inputs0
matmul_readvariableop_resource:$6-
biasadd_readvariableop_resource:6
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$6*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������6r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������6_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������6w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
E__inference_model_10_layer_call_and_return_conditional_losses_2223137

inputs"
dense_40_2223111:	$
dense_40_2223113:$,
batch_normalization_10_2223116:$,
batch_normalization_10_2223118:$,
batch_normalization_10_2223120:$,
batch_normalization_10_2223122:$"
dense_41_2223126:$6
dense_41_2223128:6"
dense_43_2223131:6
dense_43_2223133:
identity��.batch_normalization_10/StatefulPartitionedCall� dense_40/StatefulPartitionedCall� dense_41/StatefulPartitionedCall� dense_43/StatefulPartitionedCall�
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinputsdense_40_2223111dense_40_2223113*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_2222967�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0batch_normalization_10_2223116batch_normalization_10_2223118batch_normalization_10_2223120batch_normalization_10_2223122*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2222926�
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_2222987�
 dense_41/StatefulPartitionedCallStatefulPartitionedCall!re_lu_10/PartitionedCall:output:0dense_41_2223126dense_41_2223128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_2222999�
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0dense_43_2223131dense_43_2223133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_2223015x
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�M
�	
E__inference_model_10_layer_call_and_return_conditional_losses_2223378

inputs9
'dense_40_matmul_readvariableop_resource:	$6
(dense_40_biasadd_readvariableop_resource:$L
>batch_normalization_10_assignmovingavg_readvariableop_resource:$N
@batch_normalization_10_assignmovingavg_1_readvariableop_resource:$J
<batch_normalization_10_batchnorm_mul_readvariableop_resource:$F
8batch_normalization_10_batchnorm_readvariableop_resource:$9
'dense_41_matmul_readvariableop_resource:$66
(dense_41_biasadd_readvariableop_resource:69
'dense_43_matmul_readvariableop_resource:66
(dense_43_biasadd_readvariableop_resource:
identity��&batch_normalization_10/AssignMovingAvg�5batch_normalization_10/AssignMovingAvg/ReadVariableOp�(batch_normalization_10/AssignMovingAvg_1�7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_10/batchnorm/ReadVariableOp�3batch_normalization_10/batchnorm/mul/ReadVariableOp�dense_40/BiasAdd/ReadVariableOp�dense_40/MatMul/ReadVariableOp�dense_41/BiasAdd/ReadVariableOp�dense_41/MatMul/ReadVariableOp�dense_43/BiasAdd/ReadVariableOp�dense_43/MatMul/ReadVariableOp�
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:	$*
dtype0{
dense_40/MatMulMatMulinputs&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$�
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0�
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_10/moments/meanMeandense_40/BiasAdd:output:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(�
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes

:$�
0batch_normalization_10/moments/SquaredDifferenceSquaredDifferencedense_40/BiasAdd:output:04batch_normalization_10/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������$�
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(�
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 �
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 q
,batch_normalization_10/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource*
_output_shapes
:$*
dtype0�
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0*
T0*
_output_shapes
:$�
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:05batch_normalization_10/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:$�
&batch_normalization_10/AssignMovingAvgAssignSubVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_10/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource*
_output_shapes
:$*
dtype0�
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0*
T0*
_output_shapes
:$�
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:07batch_normalization_10/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:$�
(batch_normalization_10/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource0batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:$~
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:$�
3batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:0;batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$�
&batch_normalization_10/batchnorm/mul_1Muldense_40/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������$�
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:$�
/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0�
$batch_normalization_10/batchnorm/subSub7batch_normalization_10/batchnorm/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:$�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������$s
re_lu_10/ReluRelu*batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������$�
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:$6*
dtype0�
dense_41/MatMulMatMulre_lu_10/Relu:activations:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������6�
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype0�
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������6�
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes

:6*
dtype0�
dense_43/MatMulMatMuldense_41/BiasAdd:output:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_43/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_10/AssignMovingAvg6^batch_normalization_10/AssignMovingAvg/ReadVariableOp)^batch_normalization_10/AssignMovingAvg_18^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_10/batchnorm/ReadVariableOp4^batch_normalization_10/batchnorm/mul/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 2P
&batch_normalization_10/AssignMovingAvg&batch_normalization_10/AssignMovingAvg2n
5batch_normalization_10/AssignMovingAvg/ReadVariableOp5batch_normalization_10/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_10/AssignMovingAvg_1(batch_normalization_10/AssignMovingAvg_12r
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_10/batchnorm/ReadVariableOp/batch_normalization_10/batchnorm/ReadVariableOp2j
3batch_normalization_10/batchnorm/mul/ReadVariableOp3batch_normalization_10/batchnorm/mul/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
a
E__inference_re_lu_10_layer_call_and_return_conditional_losses_2222987

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������$Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
E__inference_dense_41_layer_call_and_return_conditional_losses_2222999

inputs0
matmul_readvariableop_resource:$6-
biasadd_readvariableop_resource:6
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$6*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������6r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������6_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������6w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2223496

inputs5
'assignmovingavg_readvariableop_resource:$7
)assignmovingavg_1_readvariableop_resource:$3
%batchnorm_mul_readvariableop_resource:$/
!batchnorm_readvariableop_resource:$
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:$�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������$l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:$*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:$*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:$x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:$�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:$*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:$~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:$�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������$h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:$v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������$b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������$�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������$: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�

�
*__inference_model_10_layer_call_fn_2223300

inputs
unknown:	$
	unknown_0:$
	unknown_1:$
	unknown_2:$
	unknown_3:$
	unknown_4:$
	unknown_5:$6
	unknown_6:6
	unknown_7:6
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_2223083o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_111
serving_default_input_11:0���������	<
dense_430
StatefulPartitionedCall:0���������tensorflow/serving/predict:Ν
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	gamma
 beta
!moving_mean
"moving_variance"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
f
0
1
2
 3
!4
"5
/6
07
78
89"
trackable_list_wrapper
X
0
1
2
 3
/4
05
76
87"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
>trace_0
?trace_1
@trace_2
Atrace_32�
*__inference_model_10_layer_call_fn_2223106
*__inference_model_10_layer_call_fn_2223160
*__inference_model_10_layer_call_fn_2223300
*__inference_model_10_layer_call_fn_2223325�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>trace_0z?trace_1z@trace_2zAtrace_3
�
Btrace_0
Ctrace_1
Dtrace_2
Etrace_32�
E__inference_model_10_layer_call_and_return_conditional_losses_2223022
E__inference_model_10_layer_call_and_return_conditional_losses_2223051
E__inference_model_10_layer_call_and_return_conditional_losses_2223378
E__inference_model_10_layer_call_and_return_conditional_losses_2223417�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zBtrace_0zCtrace_1zDtrace_2zEtrace_3
�B�
"__inference__wrapped_model_2222871input_11"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
F
_variables
G_iterations
H_learning_rate
I_index_dict
J
_momentums
K_velocities
L_update_step_xla"
experimentalOptimizer
,
Mserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Strace_02�
*__inference_dense_40_layer_call_fn_2223426�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0
�
Ttrace_02�
E__inference_dense_40_layer_call_and_return_conditional_losses_2223436�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0
!:	$2dense_40/kernel
:$2dense_40/bias
<
0
 1
!2
"3"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_0
[trace_12�
8__inference_batch_normalization_10_layer_call_fn_2223449
8__inference_batch_normalization_10_layer_call_fn_2223462�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0z[trace_1
�
\trace_0
]trace_12�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2223496
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2223516�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z\trace_0z]trace_1
 "
trackable_list_wrapper
*:($2batch_normalization_10/gamma
):'$2batch_normalization_10/beta
2:0$ (2"batch_normalization_10/moving_mean
6:4$ (2&batch_normalization_10/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
ctrace_02�
*__inference_re_lu_10_layer_call_fn_2223521�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zctrace_0
�
dtrace_02�
E__inference_re_lu_10_layer_call_and_return_conditional_losses_2223526�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
jtrace_02�
*__inference_dense_41_layer_call_fn_2223535�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zjtrace_0
�
ktrace_02�
E__inference_dense_41_layer_call_and_return_conditional_losses_2223545�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zktrace_0
!:$62dense_41/kernel
:62dense_41/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
qtrace_02�
*__inference_dense_43_layer_call_fn_2223554�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0
�
rtrace_02�
E__inference_dense_43_layer_call_and_return_conditional_losses_2223564�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0
!:62dense_43/kernel
:2dense_43/bias
.
!0
"1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
'
s0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_model_10_layer_call_fn_2223106input_11"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_model_10_layer_call_fn_2223160input_11"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_model_10_layer_call_fn_2223300inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_model_10_layer_call_fn_2223325inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_model_10_layer_call_and_return_conditional_losses_2223022input_11"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_model_10_layer_call_and_return_conditional_losses_2223051input_11"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_model_10_layer_call_and_return_conditional_losses_2223378inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_model_10_layer_call_and_return_conditional_losses_2223417inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
G0
t1
u2
v3
w4
x5
y6
z7
{8
|9
}10
~11
12
�13
�14
�15
�16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
Z
t0
v1
x2
z3
|4
~5
�6
�7"
trackable_list_wrapper
Z
u0
w1
y2
{3
}4
5
�6
�7"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
%__inference_signature_wrapper_2223275input_11"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_40_layer_call_fn_2223426inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_40_layer_call_and_return_conditional_losses_2223436inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_10_layer_call_fn_2223449inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_10_layer_call_fn_2223462inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2223496inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2223516inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_re_lu_10_layer_call_fn_2223521inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_re_lu_10_layer_call_and_return_conditional_losses_2223526inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_41_layer_call_fn_2223535inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_41_layer_call_and_return_conditional_losses_2223545inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_43_layer_call_fn_2223554inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_43_layer_call_and_return_conditional_losses_2223564inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
&:$	$2Adam/m/dense_40/kernel
&:$	$2Adam/v/dense_40/kernel
 :$2Adam/m/dense_40/bias
 :$2Adam/v/dense_40/bias
/:-$2#Adam/m/batch_normalization_10/gamma
/:-$2#Adam/v/batch_normalization_10/gamma
.:,$2"Adam/m/batch_normalization_10/beta
.:,$2"Adam/v/batch_normalization_10/beta
&:$$62Adam/m/dense_41/kernel
&:$$62Adam/v/dense_41/kernel
 :62Adam/m/dense_41/bias
 :62Adam/v/dense_41/bias
&:$62Adam/m/dense_43/kernel
&:$62Adam/v/dense_43/kernel
 :2Adam/m/dense_43/bias
 :2Adam/v/dense_43/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
"__inference__wrapped_model_2222871t
"! /0781�.
'�$
"�
input_11���������	
� "3�0
.
dense_43"�
dense_43����������
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2223496m!" 7�4
-�*
 �
inputs���������$
p

 
� ",�)
"�
tensor_0���������$
� �
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2223516m"! 7�4
-�*
 �
inputs���������$
p 

 
� ",�)
"�
tensor_0���������$
� �
8__inference_batch_normalization_10_layer_call_fn_2223449b!" 7�4
-�*
 �
inputs���������$
p

 
� "!�
unknown���������$�
8__inference_batch_normalization_10_layer_call_fn_2223462b"! 7�4
-�*
 �
inputs���������$
p 

 
� "!�
unknown���������$�
E__inference_dense_40_layer_call_and_return_conditional_losses_2223436c/�,
%�"
 �
inputs���������	
� ",�)
"�
tensor_0���������$
� �
*__inference_dense_40_layer_call_fn_2223426X/�,
%�"
 �
inputs���������	
� "!�
unknown���������$�
E__inference_dense_41_layer_call_and_return_conditional_losses_2223545c/0/�,
%�"
 �
inputs���������$
� ",�)
"�
tensor_0���������6
� �
*__inference_dense_41_layer_call_fn_2223535X/0/�,
%�"
 �
inputs���������$
� "!�
unknown���������6�
E__inference_dense_43_layer_call_and_return_conditional_losses_2223564c78/�,
%�"
 �
inputs���������6
� ",�)
"�
tensor_0���������
� �
*__inference_dense_43_layer_call_fn_2223554X78/�,
%�"
 �
inputs���������6
� "!�
unknown����������
E__inference_model_10_layer_call_and_return_conditional_losses_2223022u
!" /0789�6
/�,
"�
input_11���������	
p

 
� ",�)
"�
tensor_0���������
� �
E__inference_model_10_layer_call_and_return_conditional_losses_2223051u
"! /0789�6
/�,
"�
input_11���������	
p 

 
� ",�)
"�
tensor_0���������
� �
E__inference_model_10_layer_call_and_return_conditional_losses_2223378s
!" /0787�4
-�*
 �
inputs���������	
p

 
� ",�)
"�
tensor_0���������
� �
E__inference_model_10_layer_call_and_return_conditional_losses_2223417s
"! /0787�4
-�*
 �
inputs���������	
p 

 
� ",�)
"�
tensor_0���������
� �
*__inference_model_10_layer_call_fn_2223106j
!" /0789�6
/�,
"�
input_11���������	
p

 
� "!�
unknown����������
*__inference_model_10_layer_call_fn_2223160j
"! /0789�6
/�,
"�
input_11���������	
p 

 
� "!�
unknown����������
*__inference_model_10_layer_call_fn_2223300h
!" /0787�4
-�*
 �
inputs���������	
p

 
� "!�
unknown����������
*__inference_model_10_layer_call_fn_2223325h
"! /0787�4
-�*
 �
inputs���������	
p 

 
� "!�
unknown����������
E__inference_re_lu_10_layer_call_and_return_conditional_losses_2223526_/�,
%�"
 �
inputs���������$
� ",�)
"�
tensor_0���������$
� �
*__inference_re_lu_10_layer_call_fn_2223521T/�,
%�"
 �
inputs���������$
� "!�
unknown���������$�
%__inference_signature_wrapper_2223275�
"! /078=�:
� 
3�0
.
input_11"�
input_11���������	"3�0
.
dense_43"�
dense_43���������