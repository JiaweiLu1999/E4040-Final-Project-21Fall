Ê©<
É
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


DepthToSpace

input"T
output"T"	
Ttype"

block_sizeint(0":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8,

input_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameinput_conv/kernel

%input_conv/kernel/Read/ReadVariableOpReadVariableOpinput_conv/kernel*&
_output_shapes
:@*
dtype0
v
input_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameinput_conv/bias
o
#input_conv/bias/Read/ReadVariableOpReadVariableOpinput_conv/bias*
_output_shapes
:@*
dtype0

downsampler_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_namedownsampler_1/kernel

(downsampler_1/kernel/Read/ReadVariableOpReadVariableOpdownsampler_1/kernel*&
_output_shapes
:@@*
dtype0
|
downsampler_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namedownsampler_1/bias
u
&downsampler_1/bias/Read/ReadVariableOpReadVariableOpdownsampler_1/bias*
_output_shapes
:@*
dtype0

resblock_part1_1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_1_conv1/kernel

1resblock_part1_1_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_1_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_1_conv1/bias

/resblock_part1_1_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part1_1_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part1_1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_1_conv2/kernel

1resblock_part1_1_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_1_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_1_conv2/bias

/resblock_part1_1_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part1_1_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part1_2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_2_conv1/kernel

1resblock_part1_2_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_2_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_2_conv1/bias

/resblock_part1_2_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part1_2_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part1_2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_2_conv2/kernel

1resblock_part1_2_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_2_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_2_conv2/bias

/resblock_part1_2_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part1_2_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part1_3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_3_conv1/kernel

1resblock_part1_3_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_3_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_3_conv1/bias

/resblock_part1_3_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part1_3_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part1_3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_3_conv2/kernel

1resblock_part1_3_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_3_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_3_conv2/bias

/resblock_part1_3_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part1_3_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part1_4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_4_conv1/kernel

1resblock_part1_4_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_4_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_4_conv1/bias

/resblock_part1_4_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part1_4_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part1_4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part1_4_conv2/kernel

1resblock_part1_4_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part1_4_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part1_4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part1_4_conv2/bias

/resblock_part1_4_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part1_4_conv2/bias*
_output_shapes
:@*
dtype0

downsampler_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_namedownsampler_2/kernel

(downsampler_2/kernel/Read/ReadVariableOpReadVariableOpdownsampler_2/kernel*&
_output_shapes
:@@*
dtype0
|
downsampler_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namedownsampler_2/bias
u
&downsampler_2/bias/Read/ReadVariableOpReadVariableOpdownsampler_2/bias*
_output_shapes
:@*
dtype0

resblock_part2_1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_1_conv1/kernel

1resblock_part2_1_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_1_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_1_conv1/bias

/resblock_part2_1_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_1_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_1_conv2/kernel

1resblock_part2_1_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_1_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_1_conv2/bias

/resblock_part2_1_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_1_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_2_conv1/kernel

1resblock_part2_2_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_2_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_2_conv1/bias

/resblock_part2_2_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_2_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_2_conv2/kernel

1resblock_part2_2_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_2_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_2_conv2/bias

/resblock_part2_2_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_2_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_3_conv1/kernel

1resblock_part2_3_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_3_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_3_conv1/bias

/resblock_part2_3_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_3_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_3_conv2/kernel

1resblock_part2_3_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_3_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_3_conv2/bias

/resblock_part2_3_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_3_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_4_conv1/kernel

1resblock_part2_4_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_4_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_4_conv1/bias

/resblock_part2_4_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_4_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_4_conv2/kernel

1resblock_part2_4_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_4_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_4_conv2/bias

/resblock_part2_4_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_4_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_5_conv1/kernel

1resblock_part2_5_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_5_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_5_conv1/bias

/resblock_part2_5_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_5_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_5_conv2/kernel

1resblock_part2_5_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_5_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_5_conv2/bias

/resblock_part2_5_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_5_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_6_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_6_conv1/kernel

1resblock_part2_6_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_6_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_6_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_6_conv1/bias

/resblock_part2_6_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_6_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_6_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_6_conv2/kernel

1resblock_part2_6_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_6_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_6_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_6_conv2/bias

/resblock_part2_6_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_6_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_7_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_7_conv1/kernel

1resblock_part2_7_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_7_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_7_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_7_conv1/bias

/resblock_part2_7_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_7_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_7_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_7_conv2/kernel

1resblock_part2_7_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_7_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_7_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_7_conv2/bias

/resblock_part2_7_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_7_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part2_8_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_8_conv1/kernel

1resblock_part2_8_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_8_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_8_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_8_conv1/bias

/resblock_part2_8_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part2_8_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part2_8_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part2_8_conv2/kernel

1resblock_part2_8_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part2_8_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part2_8_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part2_8_conv2/bias

/resblock_part2_8_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part2_8_conv2/bias*
_output_shapes
:@*
dtype0

upsampler_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameupsampler_1/kernel

&upsampler_1/kernel/Read/ReadVariableOpReadVariableOpupsampler_1/kernel*'
_output_shapes
:@*
dtype0
y
upsampler_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameupsampler_1/bias
r
$upsampler_1/bias/Read/ReadVariableOpReadVariableOpupsampler_1/bias*
_output_shapes	
:*
dtype0

resblock_part3_1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_1_conv1/kernel

1resblock_part3_1_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_1_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_1_conv1/bias

/resblock_part3_1_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part3_1_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part3_1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_1_conv2/kernel

1resblock_part3_1_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_1_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_1_conv2/bias

/resblock_part3_1_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part3_1_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part3_2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_2_conv1/kernel

1resblock_part3_2_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_2_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_2_conv1/bias

/resblock_part3_2_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part3_2_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part3_2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_2_conv2/kernel

1resblock_part3_2_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_2_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_2_conv2/bias

/resblock_part3_2_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part3_2_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part3_3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_3_conv1/kernel

1resblock_part3_3_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_3_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_3_conv1/bias

/resblock_part3_3_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part3_3_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part3_3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_3_conv2/kernel

1resblock_part3_3_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_3_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_3_conv2/bias

/resblock_part3_3_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part3_3_conv2/bias*
_output_shapes
:@*
dtype0

resblock_part3_4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_4_conv1/kernel

1resblock_part3_4_conv1/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_4_conv1/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_4_conv1/bias

/resblock_part3_4_conv1/bias/Read/ReadVariableOpReadVariableOpresblock_part3_4_conv1/bias*
_output_shapes
:@*
dtype0

resblock_part3_4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresblock_part3_4_conv2/kernel

1resblock_part3_4_conv2/kernel/Read/ReadVariableOpReadVariableOpresblock_part3_4_conv2/kernel*&
_output_shapes
:@@*
dtype0

resblock_part3_4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameresblock_part3_4_conv2/bias

/resblock_part3_4_conv2/bias/Read/ReadVariableOpReadVariableOpresblock_part3_4_conv2/bias*
_output_shapes
:@*
dtype0

extra_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*"
shared_nameextra_conv/kernel

%extra_conv/kernel/Read/ReadVariableOpReadVariableOpextra_conv/kernel*&
_output_shapes
:@@*
dtype0
v
extra_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameextra_conv/bias
o
#extra_conv/bias/Read/ReadVariableOpReadVariableOpextra_conv/bias*
_output_shapes
:@*
dtype0

upsampler_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameupsampler_2/kernel

&upsampler_2/kernel/Read/ReadVariableOpReadVariableOpupsampler_2/kernel*'
_output_shapes
:@*
dtype0
y
upsampler_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameupsampler_2/bias
r
$upsampler_2/bias/Read/ReadVariableOpReadVariableOpupsampler_2/bias*
_output_shapes	
:*
dtype0

output_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameoutput_conv/kernel

&output_conv/kernel/Read/ReadVariableOpReadVariableOpoutput_conv/kernel*&
_output_shapes
:@*
dtype0
x
output_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameoutput_conv/bias
q
$output_conv/bias/Read/ReadVariableOpReadVariableOpoutput_conv/bias*
_output_shapes
:*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_8Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_9Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_10Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_11Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_12Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_13Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_14Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
M
Const_15Const*
_output_shapes
: *
dtype0*
valueB
 *  ?

NoOpNoOp
¯û
Const_16Const"/device:CPU:0*
_output_shapes
: *
dtype0*æú
valueÛúB×ú BÏú

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer-24
layer_with_weights-10
layer-25
layer_with_weights-11
layer-26
layer-27
layer_with_weights-12
layer-28
layer-29
layer-30
 layer_with_weights-13
 layer-31
!layer-32
"layer_with_weights-14
"layer-33
#layer-34
$layer-35
%layer_with_weights-15
%layer-36
&layer-37
'layer_with_weights-16
'layer-38
(layer-39
)layer-40
*layer_with_weights-17
*layer-41
+layer-42
,layer_with_weights-18
,layer-43
-layer-44
.layer-45
/layer_with_weights-19
/layer-46
0layer-47
1layer_with_weights-20
1layer-48
2layer-49
3layer-50
4layer_with_weights-21
4layer-51
5layer-52
6layer_with_weights-22
6layer-53
7layer-54
8layer-55
9layer_with_weights-23
9layer-56
:layer-57
;layer_with_weights-24
;layer-58
<layer-59
=layer-60
>layer_with_weights-25
>layer-61
?layer-62
@layer_with_weights-26
@layer-63
Alayer-64
Blayer-65
Clayer_with_weights-27
Clayer-66
Dlayer-67
Elayer_with_weights-28
Elayer-68
Flayer-69
Glayer_with_weights-29
Glayer-70
Hlayer-71
Ilayer-72
Jlayer_with_weights-30
Jlayer-73
Klayer-74
Llayer_with_weights-31
Llayer-75
Mlayer-76
Nlayer-77
Olayer_with_weights-32
Olayer-78
Player-79
Qlayer_with_weights-33
Qlayer-80
Rlayer-81
Slayer-82
Tlayer_with_weights-34
Tlayer-83
Ulayer-84
Vlayer_with_weights-35
Vlayer-85
Wlayer-86
Xlayer-87
Ylayer_with_weights-36
Ylayer-88
Zlayer-89
[layer_with_weights-37
[layer-90
\layer-91
]layer_with_weights-38
]layer-92
^regularization_losses
_trainable_variables
`	variables
a	keras_api
b
signatures
 
h

ckernel
dbias
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
R
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
h

mkernel
nbias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
h

skernel
tbias
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
R
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
k

}kernel
~bias
regularization_losses
trainable_variables
	variables
	keras_api

	keras_api

	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api

	keras_api

	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
 	keras_api
n
¡kernel
	¢bias
£regularization_losses
¤trainable_variables
¥	variables
¦	keras_api

§	keras_api

¨	keras_api
n
©kernel
	ªbias
«regularization_losses
¬trainable_variables
­	variables
®	keras_api
V
¯regularization_losses
°trainable_variables
±	variables
²	keras_api
n
³kernel
	´bias
µregularization_losses
¶trainable_variables
·	variables
¸	keras_api

¹	keras_api

º	keras_api
V
»regularization_losses
¼trainable_variables
½	variables
¾	keras_api
n
¿kernel
	Àbias
Áregularization_losses
Âtrainable_variables
Ã	variables
Ä	keras_api
n
Åkernel
	Æbias
Çregularization_losses
Ètrainable_variables
É	variables
Ê	keras_api
V
Ëregularization_losses
Ìtrainable_variables
Í	variables
Î	keras_api
n
Ïkernel
	Ðbias
Ñregularization_losses
Òtrainable_variables
Ó	variables
Ô	keras_api

Õ	keras_api

Ö	keras_api
n
×kernel
	Øbias
Ùregularization_losses
Útrainable_variables
Û	variables
Ü	keras_api
V
Ýregularization_losses
Þtrainable_variables
ß	variables
à	keras_api
n
ákernel
	âbias
ãregularization_losses
ätrainable_variables
å	variables
æ	keras_api

ç	keras_api

è	keras_api
n
ékernel
	êbias
ëregularization_losses
ìtrainable_variables
í	variables
î	keras_api
V
ïregularization_losses
ðtrainable_variables
ñ	variables
ò	keras_api
n
ókernel
	ôbias
õregularization_losses
ötrainable_variables
÷	variables
ø	keras_api

ù	keras_api

ú	keras_api
n
ûkernel
	übias
ýregularization_losses
þtrainable_variables
ÿ	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api

	keras_api

	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api

	keras_api

	keras_api
n
kernel
	 bias
¡regularization_losses
¢trainable_variables
£	variables
¤	keras_api
V
¥regularization_losses
¦trainable_variables
§	variables
¨	keras_api
n
©kernel
	ªbias
«regularization_losses
¬trainable_variables
­	variables
®	keras_api

¯	keras_api

°	keras_api
n
±kernel
	²bias
³regularization_losses
´trainable_variables
µ	variables
¶	keras_api
V
·regularization_losses
¸trainable_variables
¹	variables
º	keras_api
n
»kernel
	¼bias
½regularization_losses
¾trainable_variables
¿	variables
À	keras_api

Á	keras_api

Â	keras_api
n
Ãkernel
	Äbias
Åregularization_losses
Ætrainable_variables
Ç	variables
È	keras_api
V
Éregularization_losses
Êtrainable_variables
Ë	variables
Ì	keras_api
n
Íkernel
	Îbias
Ïregularization_losses
Ðtrainable_variables
Ñ	variables
Ò	keras_api

Ó	keras_api

Ô	keras_api
n
Õkernel
	Öbias
×regularization_losses
Øtrainable_variables
Ù	variables
Ú	keras_api

Û	keras_api
n
Ükernel
	Ýbias
Þregularization_losses
ßtrainable_variables
à	variables
á	keras_api
V
âregularization_losses
ãtrainable_variables
ä	variables
å	keras_api
n
ækernel
	çbias
èregularization_losses
étrainable_variables
ê	variables
ë	keras_api

ì	keras_api

í	keras_api
n
îkernel
	ïbias
ðregularization_losses
ñtrainable_variables
ò	variables
ó	keras_api
V
ôregularization_losses
õtrainable_variables
ö	variables
÷	keras_api
n
økernel
	ùbias
úregularization_losses
ûtrainable_variables
ü	variables
ý	keras_api

þ	keras_api

ÿ	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api

	keras_api

	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
n
kernel
	bias
regularization_losses
trainable_variables
 	variables
¡	keras_api

¢	keras_api

£	keras_api
n
¤kernel
	¥bias
¦regularization_losses
§trainable_variables
¨	variables
©	keras_api

ª	keras_api
n
«kernel
	¬bias
­regularization_losses
®trainable_variables
¯	variables
°	keras_api

±	keras_api
n
²kernel
	³bias
´regularization_losses
µtrainable_variables
¶	variables
·	keras_api
 
¬
c0
d1
m2
n3
s4
t5
}6
~7
8
9
10
11
12
13
¡14
¢15
©16
ª17
³18
´19
¿20
À21
Å22
Æ23
Ï24
Ð25
×26
Ø27
á28
â29
é30
ê31
ó32
ô33
û34
ü35
36
37
38
39
40
41
42
 43
©44
ª45
±46
²47
»48
¼49
Ã50
Ä51
Í52
Î53
Õ54
Ö55
Ü56
Ý57
æ58
ç59
î60
ï61
ø62
ù63
64
65
66
67
68
69
70
71
¤72
¥73
«74
¬75
²76
³77
¬
c0
d1
m2
n3
s4
t5
}6
~7
8
9
10
11
12
13
¡14
¢15
©16
ª17
³18
´19
¿20
À21
Å22
Æ23
Ï24
Ð25
×26
Ø27
á28
â29
é30
ê31
ó32
ô33
û34
ü35
36
37
38
39
40
41
42
 43
©44
ª45
±46
²47
»48
¼49
Ã50
Ä51
Í52
Î53
Õ54
Ö55
Ü56
Ý57
æ58
ç59
î60
ï61
ø62
ù63
64
65
66
67
68
69
70
71
¤72
¥73
«74
¬75
²76
³77
²
^regularization_losses
¸layers
_trainable_variables
`	variables
¹metrics
ºnon_trainable_variables
»layer_metrics
 ¼layer_regularization_losses
 
][
VARIABLE_VALUEinput_conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEinput_conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

c0
d1

c0
d1
²
eregularization_losses
½layers
ftrainable_variables
g	variables
¾metrics
¿non_trainable_variables
Àlayer_metrics
 Álayer_regularization_losses
 
 
 
²
iregularization_losses
Âlayers
jtrainable_variables
k	variables
Ãmetrics
Änon_trainable_variables
Ålayer_metrics
 Ælayer_regularization_losses
`^
VARIABLE_VALUEdownsampler_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdownsampler_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

m0
n1
²
oregularization_losses
Çlayers
ptrainable_variables
q	variables
Èmetrics
Énon_trainable_variables
Êlayer_metrics
 Ëlayer_regularization_losses
ig
VARIABLE_VALUEresblock_part1_1_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_1_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

s0
t1

s0
t1
²
uregularization_losses
Ìlayers
vtrainable_variables
w	variables
Ímetrics
Înon_trainable_variables
Ïlayer_metrics
 Ðlayer_regularization_losses
 
 
 
²
yregularization_losses
Ñlayers
ztrainable_variables
{	variables
Òmetrics
Ónon_trainable_variables
Ôlayer_metrics
 Õlayer_regularization_losses
ig
VARIABLE_VALUEresblock_part1_1_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_1_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

}0
~1

}0
~1
´
regularization_losses
Ölayers
trainable_variables
	variables
×metrics
Ønon_trainable_variables
Ùlayer_metrics
 Úlayer_regularization_losses
 
 
ig
VARIABLE_VALUEresblock_part1_2_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_2_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
Ûlayers
trainable_variables
	variables
Ümetrics
Ýnon_trainable_variables
Þlayer_metrics
 ßlayer_regularization_losses
 
 
 
µ
regularization_losses
àlayers
trainable_variables
	variables
ámetrics
ânon_trainable_variables
ãlayer_metrics
 älayer_regularization_losses
ig
VARIABLE_VALUEresblock_part1_2_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_2_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
ålayers
trainable_variables
	variables
æmetrics
çnon_trainable_variables
èlayer_metrics
 élayer_regularization_losses
 
 
ig
VARIABLE_VALUEresblock_part1_3_conv1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_3_conv1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
êlayers
trainable_variables
	variables
ëmetrics
ìnon_trainable_variables
ílayer_metrics
 îlayer_regularization_losses
 
 
 
µ
regularization_losses
ïlayers
trainable_variables
	variables
ðmetrics
ñnon_trainable_variables
òlayer_metrics
 ólayer_regularization_losses
ig
VARIABLE_VALUEresblock_part1_3_conv2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_3_conv2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

¡0
¢1

¡0
¢1
µ
£regularization_losses
ôlayers
¤trainable_variables
¥	variables
õmetrics
önon_trainable_variables
÷layer_metrics
 ølayer_regularization_losses
 
 
ig
VARIABLE_VALUEresblock_part1_4_conv1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_4_conv1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

©0
ª1

©0
ª1
µ
«regularization_losses
ùlayers
¬trainable_variables
­	variables
úmetrics
ûnon_trainable_variables
ülayer_metrics
 ýlayer_regularization_losses
 
 
 
µ
¯regularization_losses
þlayers
°trainable_variables
±	variables
ÿmetrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
ig
VARIABLE_VALUEresblock_part1_4_conv2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_4_conv2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

³0
´1

³0
´1
µ
µregularization_losses
layers
¶trainable_variables
·	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
 
 
 
µ
»regularization_losses
layers
¼trainable_variables
½	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
a_
VARIABLE_VALUEdownsampler_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdownsampler_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

¿0
À1

¿0
À1
µ
Áregularization_losses
layers
Âtrainable_variables
Ã	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
jh
VARIABLE_VALUEresblock_part2_1_conv1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_1_conv1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Å0
Æ1

Å0
Æ1
µ
Çregularization_losses
layers
Ètrainable_variables
É	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
 
µ
Ëregularization_losses
layers
Ìtrainable_variables
Í	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
jh
VARIABLE_VALUEresblock_part2_1_conv2/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_1_conv2/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ï0
Ð1

Ï0
Ð1
µ
Ñregularization_losses
layers
Òtrainable_variables
Ó	variables
metrics
non_trainable_variables
layer_metrics
  layer_regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_2_conv1/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_2_conv1/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

×0
Ø1

×0
Ø1
µ
Ùregularization_losses
¡layers
Útrainable_variables
Û	variables
¢metrics
£non_trainable_variables
¤layer_metrics
 ¥layer_regularization_losses
 
 
 
µ
Ýregularization_losses
¦layers
Þtrainable_variables
ß	variables
§metrics
¨non_trainable_variables
©layer_metrics
 ªlayer_regularization_losses
jh
VARIABLE_VALUEresblock_part2_2_conv2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_2_conv2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

á0
â1

á0
â1
µ
ãregularization_losses
«layers
ätrainable_variables
å	variables
¬metrics
­non_trainable_variables
®layer_metrics
 ¯layer_regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_3_conv1/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_3_conv1/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

é0
ê1

é0
ê1
µ
ëregularization_losses
°layers
ìtrainable_variables
í	variables
±metrics
²non_trainable_variables
³layer_metrics
 ´layer_regularization_losses
 
 
 
µ
ïregularization_losses
µlayers
ðtrainable_variables
ñ	variables
¶metrics
·non_trainable_variables
¸layer_metrics
 ¹layer_regularization_losses
jh
VARIABLE_VALUEresblock_part2_3_conv2/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_3_conv2/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

ó0
ô1

ó0
ô1
µ
õregularization_losses
ºlayers
ötrainable_variables
÷	variables
»metrics
¼non_trainable_variables
½layer_metrics
 ¾layer_regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_4_conv1/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_4_conv1/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

û0
ü1

û0
ü1
µ
ýregularization_losses
¿layers
þtrainable_variables
ÿ	variables
Àmetrics
Ánon_trainable_variables
Âlayer_metrics
 Ãlayer_regularization_losses
 
 
 
µ
regularization_losses
Älayers
trainable_variables
	variables
Åmetrics
Ænon_trainable_variables
Çlayer_metrics
 Èlayer_regularization_losses
jh
VARIABLE_VALUEresblock_part2_4_conv2/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_4_conv2/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
Élayers
trainable_variables
	variables
Êmetrics
Ënon_trainable_variables
Ìlayer_metrics
 Ílayer_regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_5_conv1/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_5_conv1/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
Îlayers
trainable_variables
	variables
Ïmetrics
Ðnon_trainable_variables
Ñlayer_metrics
 Òlayer_regularization_losses
 
 
 
µ
regularization_losses
Ólayers
trainable_variables
	variables
Ômetrics
Õnon_trainable_variables
Ölayer_metrics
 ×layer_regularization_losses
jh
VARIABLE_VALUEresblock_part2_5_conv2/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_5_conv2/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
Ølayers
trainable_variables
	variables
Ùmetrics
Únon_trainable_variables
Ûlayer_metrics
 Ülayer_regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_6_conv1/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_6_conv1/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
µ
¡regularization_losses
Ýlayers
¢trainable_variables
£	variables
Þmetrics
ßnon_trainable_variables
àlayer_metrics
 álayer_regularization_losses
 
 
 
µ
¥regularization_losses
âlayers
¦trainable_variables
§	variables
ãmetrics
änon_trainable_variables
ålayer_metrics
 ælayer_regularization_losses
jh
VARIABLE_VALUEresblock_part2_6_conv2/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_6_conv2/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE
 

©0
ª1

©0
ª1
µ
«regularization_losses
çlayers
¬trainable_variables
­	variables
èmetrics
énon_trainable_variables
êlayer_metrics
 ëlayer_regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_7_conv1/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_7_conv1/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE
 

±0
²1

±0
²1
µ
³regularization_losses
ìlayers
´trainable_variables
µ	variables
ímetrics
înon_trainable_variables
ïlayer_metrics
 ðlayer_regularization_losses
 
 
 
µ
·regularization_losses
ñlayers
¸trainable_variables
¹	variables
òmetrics
ónon_trainable_variables
ôlayer_metrics
 õlayer_regularization_losses
jh
VARIABLE_VALUEresblock_part2_7_conv2/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_7_conv2/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE
 

»0
¼1

»0
¼1
µ
½regularization_losses
ölayers
¾trainable_variables
¿	variables
÷metrics
ønon_trainable_variables
ùlayer_metrics
 úlayer_regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_8_conv1/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_8_conv1/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ã0
Ä1

Ã0
Ä1
µ
Åregularization_losses
ûlayers
Ætrainable_variables
Ç	variables
ümetrics
ýnon_trainable_variables
þlayer_metrics
 ÿlayer_regularization_losses
 
 
 
µ
Éregularization_losses
layers
Êtrainable_variables
Ë	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
jh
VARIABLE_VALUEresblock_part2_8_conv2/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_8_conv2/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Í0
Î1

Í0
Î1
µ
Ïregularization_losses
layers
Ðtrainable_variables
Ñ	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
_]
VARIABLE_VALUEupsampler_1/kernel7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEupsampler_1/bias5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Õ0
Ö1

Õ0
Ö1
µ
×regularization_losses
layers
Øtrainable_variables
Ù	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
jh
VARIABLE_VALUEresblock_part3_1_conv1/kernel7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_1_conv1/bias5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ü0
Ý1

Ü0
Ý1
µ
Þregularization_losses
layers
ßtrainable_variables
à	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
 
µ
âregularization_losses
layers
ãtrainable_variables
ä	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
jh
VARIABLE_VALUEresblock_part3_1_conv2/kernel7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_1_conv2/bias5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUE
 

æ0
ç1

æ0
ç1
µ
èregularization_losses
layers
étrainable_variables
ê	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part3_2_conv1/kernel7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_2_conv1/bias5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUE
 

î0
ï1

î0
ï1
µ
ðregularization_losses
layers
ñtrainable_variables
ò	variables
metrics
 non_trainable_variables
¡layer_metrics
 ¢layer_regularization_losses
 
 
 
µ
ôregularization_losses
£layers
õtrainable_variables
ö	variables
¤metrics
¥non_trainable_variables
¦layer_metrics
 §layer_regularization_losses
jh
VARIABLE_VALUEresblock_part3_2_conv2/kernel7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_2_conv2/bias5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUE
 

ø0
ù1

ø0
ù1
µ
úregularization_losses
¨layers
ûtrainable_variables
ü	variables
©metrics
ªnon_trainable_variables
«layer_metrics
 ¬layer_regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part3_3_conv1/kernel7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_3_conv1/bias5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
­layers
trainable_variables
	variables
®metrics
¯non_trainable_variables
°layer_metrics
 ±layer_regularization_losses
 
 
 
µ
regularization_losses
²layers
trainable_variables
	variables
³metrics
´non_trainable_variables
µlayer_metrics
 ¶layer_regularization_losses
jh
VARIABLE_VALUEresblock_part3_3_conv2/kernel7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_3_conv2/bias5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
·layers
trainable_variables
	variables
¸metrics
¹non_trainable_variables
ºlayer_metrics
 »layer_regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part3_4_conv1/kernel7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_4_conv1/bias5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
¼layers
trainable_variables
	variables
½metrics
¾non_trainable_variables
¿layer_metrics
 Àlayer_regularization_losses
 
 
 
µ
regularization_losses
Álayers
trainable_variables
	variables
Âmetrics
Ãnon_trainable_variables
Älayer_metrics
 Ålayer_regularization_losses
jh
VARIABLE_VALUEresblock_part3_4_conv2/kernel7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_4_conv2/bias5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
regularization_losses
Ælayers
trainable_variables
 	variables
Çmetrics
Ènon_trainable_variables
Élayer_metrics
 Êlayer_regularization_losses
 
 
^\
VARIABLE_VALUEextra_conv/kernel7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEextra_conv/bias5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUE
 

¤0
¥1

¤0
¥1
µ
¦regularization_losses
Ëlayers
§trainable_variables
¨	variables
Ìmetrics
Ínon_trainable_variables
Îlayer_metrics
 Ïlayer_regularization_losses
 
_]
VARIABLE_VALUEupsampler_2/kernel7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEupsampler_2/bias5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUE
 

«0
¬1

«0
¬1
µ
­regularization_losses
Ðlayers
®trainable_variables
¯	variables
Ñmetrics
Ònon_trainable_variables
Ólayer_metrics
 Ôlayer_regularization_losses
 
_]
VARIABLE_VALUEoutput_conv/kernel7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_conv/bias5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUE
 

²0
³1

²0
³1
µ
´regularization_losses
Õlayers
µtrainable_variables
¶	variables
Ömetrics
×non_trainable_variables
Ølayer_metrics
 Ùlayer_regularization_losses
Þ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
W86
X87
Y88
Z89
[90
\91
]92
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
 
 
 

serving_default_input_layerPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerinput_conv/kernelinput_conv/biasdownsampler_1/kerneldownsampler_1/biasresblock_part1_1_conv1/kernelresblock_part1_1_conv1/biasresblock_part1_1_conv2/kernelresblock_part1_1_conv2/biasConstresblock_part1_2_conv1/kernelresblock_part1_2_conv1/biasresblock_part1_2_conv2/kernelresblock_part1_2_conv2/biasConst_1resblock_part1_3_conv1/kernelresblock_part1_3_conv1/biasresblock_part1_3_conv2/kernelresblock_part1_3_conv2/biasConst_2resblock_part1_4_conv1/kernelresblock_part1_4_conv1/biasresblock_part1_4_conv2/kernelresblock_part1_4_conv2/biasConst_3downsampler_2/kerneldownsampler_2/biasresblock_part2_1_conv1/kernelresblock_part2_1_conv1/biasresblock_part2_1_conv2/kernelresblock_part2_1_conv2/biasConst_4resblock_part2_2_conv1/kernelresblock_part2_2_conv1/biasresblock_part2_2_conv2/kernelresblock_part2_2_conv2/biasConst_5resblock_part2_3_conv1/kernelresblock_part2_3_conv1/biasresblock_part2_3_conv2/kernelresblock_part2_3_conv2/biasConst_6resblock_part2_4_conv1/kernelresblock_part2_4_conv1/biasresblock_part2_4_conv2/kernelresblock_part2_4_conv2/biasConst_7resblock_part2_5_conv1/kernelresblock_part2_5_conv1/biasresblock_part2_5_conv2/kernelresblock_part2_5_conv2/biasConst_8resblock_part2_6_conv1/kernelresblock_part2_6_conv1/biasresblock_part2_6_conv2/kernelresblock_part2_6_conv2/biasConst_9resblock_part2_7_conv1/kernelresblock_part2_7_conv1/biasresblock_part2_7_conv2/kernelresblock_part2_7_conv2/biasConst_10resblock_part2_8_conv1/kernelresblock_part2_8_conv1/biasresblock_part2_8_conv2/kernelresblock_part2_8_conv2/biasConst_11upsampler_1/kernelupsampler_1/biasresblock_part3_1_conv1/kernelresblock_part3_1_conv1/biasresblock_part3_1_conv2/kernelresblock_part3_1_conv2/biasConst_12resblock_part3_2_conv1/kernelresblock_part3_2_conv1/biasresblock_part3_2_conv2/kernelresblock_part3_2_conv2/biasConst_13resblock_part3_3_conv1/kernelresblock_part3_3_conv1/biasresblock_part3_3_conv2/kernelresblock_part3_3_conv2/biasConst_14resblock_part3_4_conv1/kernelresblock_part3_4_conv1/biasresblock_part3_4_conv2/kernelresblock_part3_4_conv2/biasConst_15extra_conv/kernelextra_conv/biasupsampler_2/kernelupsampler_2/biasoutput_conv/kerneloutput_conv/bias*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_4750
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%input_conv/kernel/Read/ReadVariableOp#input_conv/bias/Read/ReadVariableOp(downsampler_1/kernel/Read/ReadVariableOp&downsampler_1/bias/Read/ReadVariableOp1resblock_part1_1_conv1/kernel/Read/ReadVariableOp/resblock_part1_1_conv1/bias/Read/ReadVariableOp1resblock_part1_1_conv2/kernel/Read/ReadVariableOp/resblock_part1_1_conv2/bias/Read/ReadVariableOp1resblock_part1_2_conv1/kernel/Read/ReadVariableOp/resblock_part1_2_conv1/bias/Read/ReadVariableOp1resblock_part1_2_conv2/kernel/Read/ReadVariableOp/resblock_part1_2_conv2/bias/Read/ReadVariableOp1resblock_part1_3_conv1/kernel/Read/ReadVariableOp/resblock_part1_3_conv1/bias/Read/ReadVariableOp1resblock_part1_3_conv2/kernel/Read/ReadVariableOp/resblock_part1_3_conv2/bias/Read/ReadVariableOp1resblock_part1_4_conv1/kernel/Read/ReadVariableOp/resblock_part1_4_conv1/bias/Read/ReadVariableOp1resblock_part1_4_conv2/kernel/Read/ReadVariableOp/resblock_part1_4_conv2/bias/Read/ReadVariableOp(downsampler_2/kernel/Read/ReadVariableOp&downsampler_2/bias/Read/ReadVariableOp1resblock_part2_1_conv1/kernel/Read/ReadVariableOp/resblock_part2_1_conv1/bias/Read/ReadVariableOp1resblock_part2_1_conv2/kernel/Read/ReadVariableOp/resblock_part2_1_conv2/bias/Read/ReadVariableOp1resblock_part2_2_conv1/kernel/Read/ReadVariableOp/resblock_part2_2_conv1/bias/Read/ReadVariableOp1resblock_part2_2_conv2/kernel/Read/ReadVariableOp/resblock_part2_2_conv2/bias/Read/ReadVariableOp1resblock_part2_3_conv1/kernel/Read/ReadVariableOp/resblock_part2_3_conv1/bias/Read/ReadVariableOp1resblock_part2_3_conv2/kernel/Read/ReadVariableOp/resblock_part2_3_conv2/bias/Read/ReadVariableOp1resblock_part2_4_conv1/kernel/Read/ReadVariableOp/resblock_part2_4_conv1/bias/Read/ReadVariableOp1resblock_part2_4_conv2/kernel/Read/ReadVariableOp/resblock_part2_4_conv2/bias/Read/ReadVariableOp1resblock_part2_5_conv1/kernel/Read/ReadVariableOp/resblock_part2_5_conv1/bias/Read/ReadVariableOp1resblock_part2_5_conv2/kernel/Read/ReadVariableOp/resblock_part2_5_conv2/bias/Read/ReadVariableOp1resblock_part2_6_conv1/kernel/Read/ReadVariableOp/resblock_part2_6_conv1/bias/Read/ReadVariableOp1resblock_part2_6_conv2/kernel/Read/ReadVariableOp/resblock_part2_6_conv2/bias/Read/ReadVariableOp1resblock_part2_7_conv1/kernel/Read/ReadVariableOp/resblock_part2_7_conv1/bias/Read/ReadVariableOp1resblock_part2_7_conv2/kernel/Read/ReadVariableOp/resblock_part2_7_conv2/bias/Read/ReadVariableOp1resblock_part2_8_conv1/kernel/Read/ReadVariableOp/resblock_part2_8_conv1/bias/Read/ReadVariableOp1resblock_part2_8_conv2/kernel/Read/ReadVariableOp/resblock_part2_8_conv2/bias/Read/ReadVariableOp&upsampler_1/kernel/Read/ReadVariableOp$upsampler_1/bias/Read/ReadVariableOp1resblock_part3_1_conv1/kernel/Read/ReadVariableOp/resblock_part3_1_conv1/bias/Read/ReadVariableOp1resblock_part3_1_conv2/kernel/Read/ReadVariableOp/resblock_part3_1_conv2/bias/Read/ReadVariableOp1resblock_part3_2_conv1/kernel/Read/ReadVariableOp/resblock_part3_2_conv1/bias/Read/ReadVariableOp1resblock_part3_2_conv2/kernel/Read/ReadVariableOp/resblock_part3_2_conv2/bias/Read/ReadVariableOp1resblock_part3_3_conv1/kernel/Read/ReadVariableOp/resblock_part3_3_conv1/bias/Read/ReadVariableOp1resblock_part3_3_conv2/kernel/Read/ReadVariableOp/resblock_part3_3_conv2/bias/Read/ReadVariableOp1resblock_part3_4_conv1/kernel/Read/ReadVariableOp/resblock_part3_4_conv1/bias/Read/ReadVariableOp1resblock_part3_4_conv2/kernel/Read/ReadVariableOp/resblock_part3_4_conv2/bias/Read/ReadVariableOp%extra_conv/kernel/Read/ReadVariableOp#extra_conv/bias/Read/ReadVariableOp&upsampler_2/kernel/Read/ReadVariableOp$upsampler_2/bias/Read/ReadVariableOp&output_conv/kernel/Read/ReadVariableOp$output_conv/bias/Read/ReadVariableOpConst_16*[
TinT
R2P*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_6928
ó
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_conv/kernelinput_conv/biasdownsampler_1/kerneldownsampler_1/biasresblock_part1_1_conv1/kernelresblock_part1_1_conv1/biasresblock_part1_1_conv2/kernelresblock_part1_1_conv2/biasresblock_part1_2_conv1/kernelresblock_part1_2_conv1/biasresblock_part1_2_conv2/kernelresblock_part1_2_conv2/biasresblock_part1_3_conv1/kernelresblock_part1_3_conv1/biasresblock_part1_3_conv2/kernelresblock_part1_3_conv2/biasresblock_part1_4_conv1/kernelresblock_part1_4_conv1/biasresblock_part1_4_conv2/kernelresblock_part1_4_conv2/biasdownsampler_2/kerneldownsampler_2/biasresblock_part2_1_conv1/kernelresblock_part2_1_conv1/biasresblock_part2_1_conv2/kernelresblock_part2_1_conv2/biasresblock_part2_2_conv1/kernelresblock_part2_2_conv1/biasresblock_part2_2_conv2/kernelresblock_part2_2_conv2/biasresblock_part2_3_conv1/kernelresblock_part2_3_conv1/biasresblock_part2_3_conv2/kernelresblock_part2_3_conv2/biasresblock_part2_4_conv1/kernelresblock_part2_4_conv1/biasresblock_part2_4_conv2/kernelresblock_part2_4_conv2/biasresblock_part2_5_conv1/kernelresblock_part2_5_conv1/biasresblock_part2_5_conv2/kernelresblock_part2_5_conv2/biasresblock_part2_6_conv1/kernelresblock_part2_6_conv1/biasresblock_part2_6_conv2/kernelresblock_part2_6_conv2/biasresblock_part2_7_conv1/kernelresblock_part2_7_conv1/biasresblock_part2_7_conv2/kernelresblock_part2_7_conv2/biasresblock_part2_8_conv1/kernelresblock_part2_8_conv1/biasresblock_part2_8_conv2/kernelresblock_part2_8_conv2/biasupsampler_1/kernelupsampler_1/biasresblock_part3_1_conv1/kernelresblock_part3_1_conv1/biasresblock_part3_1_conv2/kernelresblock_part3_1_conv2/biasresblock_part3_2_conv1/kernelresblock_part3_2_conv1/biasresblock_part3_2_conv2/kernelresblock_part3_2_conv2/biasresblock_part3_3_conv1/kernelresblock_part3_3_conv1/biasresblock_part3_3_conv2/kernelresblock_part3_3_conv2/biasresblock_part3_4_conv1/kernelresblock_part3_4_conv1/biasresblock_part3_4_conv2/kernelresblock_part3_4_conv2/biasextra_conv/kernelextra_conv/biasupsampler_2/kernelupsampler_2/biasoutput_conv/kerneloutput_conv/bias*Z
TinS
Q2O*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_7172¤(


5__inference_resblock_part2_6_conv2_layer_call_fn_6291

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_28292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_5_conv1_layer_call_fn_6214

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_27222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_3_conv1_layer_call_fn_6521

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_31572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_2625

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
ô"
¼
+__inference_ssi_res_unet_layer_call_fn_4094
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_39032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
¤

é
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_6157

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Ô
·C
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5368

inputs-
)input_conv_conv2d_readvariableop_resource.
*input_conv_biasadd_readvariableop_resource0
,downsampler_1_conv2d_readvariableop_resource1
-downsampler_1_biasadd_readvariableop_resource9
5resblock_part1_1_conv1_conv2d_readvariableop_resource:
6resblock_part1_1_conv1_biasadd_readvariableop_resource9
5resblock_part1_1_conv2_conv2d_readvariableop_resource:
6resblock_part1_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_mul_x9
5resblock_part1_2_conv1_conv2d_readvariableop_resource:
6resblock_part1_2_conv1_biasadd_readvariableop_resource9
5resblock_part1_2_conv2_conv2d_readvariableop_resource:
6resblock_part1_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_1_mul_x9
5resblock_part1_3_conv1_conv2d_readvariableop_resource:
6resblock_part1_3_conv1_biasadd_readvariableop_resource9
5resblock_part1_3_conv2_conv2d_readvariableop_resource:
6resblock_part1_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_2_mul_x9
5resblock_part1_4_conv1_conv2d_readvariableop_resource:
6resblock_part1_4_conv1_biasadd_readvariableop_resource9
5resblock_part1_4_conv2_conv2d_readvariableop_resource:
6resblock_part1_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_3_mul_x0
,downsampler_2_conv2d_readvariableop_resource1
-downsampler_2_biasadd_readvariableop_resource9
5resblock_part2_1_conv1_conv2d_readvariableop_resource:
6resblock_part2_1_conv1_biasadd_readvariableop_resource9
5resblock_part2_1_conv2_conv2d_readvariableop_resource:
6resblock_part2_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_4_mul_x9
5resblock_part2_2_conv1_conv2d_readvariableop_resource:
6resblock_part2_2_conv1_biasadd_readvariableop_resource9
5resblock_part2_2_conv2_conv2d_readvariableop_resource:
6resblock_part2_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_5_mul_x9
5resblock_part2_3_conv1_conv2d_readvariableop_resource:
6resblock_part2_3_conv1_biasadd_readvariableop_resource9
5resblock_part2_3_conv2_conv2d_readvariableop_resource:
6resblock_part2_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_6_mul_x9
5resblock_part2_4_conv1_conv2d_readvariableop_resource:
6resblock_part2_4_conv1_biasadd_readvariableop_resource9
5resblock_part2_4_conv2_conv2d_readvariableop_resource:
6resblock_part2_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_7_mul_x9
5resblock_part2_5_conv1_conv2d_readvariableop_resource:
6resblock_part2_5_conv1_biasadd_readvariableop_resource9
5resblock_part2_5_conv2_conv2d_readvariableop_resource:
6resblock_part2_5_conv2_biasadd_readvariableop_resource
tf_math_multiply_8_mul_x9
5resblock_part2_6_conv1_conv2d_readvariableop_resource:
6resblock_part2_6_conv1_biasadd_readvariableop_resource9
5resblock_part2_6_conv2_conv2d_readvariableop_resource:
6resblock_part2_6_conv2_biasadd_readvariableop_resource
tf_math_multiply_9_mul_x9
5resblock_part2_7_conv1_conv2d_readvariableop_resource:
6resblock_part2_7_conv1_biasadd_readvariableop_resource9
5resblock_part2_7_conv2_conv2d_readvariableop_resource:
6resblock_part2_7_conv2_biasadd_readvariableop_resource
tf_math_multiply_10_mul_x9
5resblock_part2_8_conv1_conv2d_readvariableop_resource:
6resblock_part2_8_conv1_biasadd_readvariableop_resource9
5resblock_part2_8_conv2_conv2d_readvariableop_resource:
6resblock_part2_8_conv2_biasadd_readvariableop_resource
tf_math_multiply_11_mul_x.
*upsampler_1_conv2d_readvariableop_resource/
+upsampler_1_biasadd_readvariableop_resource9
5resblock_part3_1_conv1_conv2d_readvariableop_resource:
6resblock_part3_1_conv1_biasadd_readvariableop_resource9
5resblock_part3_1_conv2_conv2d_readvariableop_resource:
6resblock_part3_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_12_mul_x9
5resblock_part3_2_conv1_conv2d_readvariableop_resource:
6resblock_part3_2_conv1_biasadd_readvariableop_resource9
5resblock_part3_2_conv2_conv2d_readvariableop_resource:
6resblock_part3_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_13_mul_x9
5resblock_part3_3_conv1_conv2d_readvariableop_resource:
6resblock_part3_3_conv1_biasadd_readvariableop_resource9
5resblock_part3_3_conv2_conv2d_readvariableop_resource:
6resblock_part3_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_14_mul_x9
5resblock_part3_4_conv1_conv2d_readvariableop_resource:
6resblock_part3_4_conv1_biasadd_readvariableop_resource9
5resblock_part3_4_conv2_conv2d_readvariableop_resource:
6resblock_part3_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_15_mul_x-
)extra_conv_conv2d_readvariableop_resource.
*extra_conv_biasadd_readvariableop_resource.
*upsampler_2_conv2d_readvariableop_resource/
+upsampler_2_biasadd_readvariableop_resource.
*output_conv_conv2d_readvariableop_resource/
+output_conv_biasadd_readvariableop_resource
identity¢$downsampler_1/BiasAdd/ReadVariableOp¢#downsampler_1/Conv2D/ReadVariableOp¢$downsampler_2/BiasAdd/ReadVariableOp¢#downsampler_2/Conv2D/ReadVariableOp¢!extra_conv/BiasAdd/ReadVariableOp¢ extra_conv/Conv2D/ReadVariableOp¢!input_conv/BiasAdd/ReadVariableOp¢ input_conv/Conv2D/ReadVariableOp¢"output_conv/BiasAdd/ReadVariableOp¢!output_conv/Conv2D/ReadVariableOp¢-resblock_part1_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_1_conv1/Conv2D/ReadVariableOp¢-resblock_part1_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_1_conv2/Conv2D/ReadVariableOp¢-resblock_part1_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_2_conv1/Conv2D/ReadVariableOp¢-resblock_part1_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_2_conv2/Conv2D/ReadVariableOp¢-resblock_part1_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_3_conv1/Conv2D/ReadVariableOp¢-resblock_part1_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_3_conv2/Conv2D/ReadVariableOp¢-resblock_part1_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_4_conv1/Conv2D/ReadVariableOp¢-resblock_part1_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_4_conv2/Conv2D/ReadVariableOp¢-resblock_part2_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_1_conv1/Conv2D/ReadVariableOp¢-resblock_part2_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_1_conv2/Conv2D/ReadVariableOp¢-resblock_part2_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_2_conv1/Conv2D/ReadVariableOp¢-resblock_part2_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_2_conv2/Conv2D/ReadVariableOp¢-resblock_part2_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_3_conv1/Conv2D/ReadVariableOp¢-resblock_part2_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_3_conv2/Conv2D/ReadVariableOp¢-resblock_part2_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_4_conv1/Conv2D/ReadVariableOp¢-resblock_part2_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_4_conv2/Conv2D/ReadVariableOp¢-resblock_part2_5_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_5_conv1/Conv2D/ReadVariableOp¢-resblock_part2_5_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_5_conv2/Conv2D/ReadVariableOp¢-resblock_part2_6_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_6_conv1/Conv2D/ReadVariableOp¢-resblock_part2_6_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_6_conv2/Conv2D/ReadVariableOp¢-resblock_part2_7_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_7_conv1/Conv2D/ReadVariableOp¢-resblock_part2_7_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_7_conv2/Conv2D/ReadVariableOp¢-resblock_part2_8_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_8_conv1/Conv2D/ReadVariableOp¢-resblock_part2_8_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_8_conv2/Conv2D/ReadVariableOp¢-resblock_part3_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_1_conv1/Conv2D/ReadVariableOp¢-resblock_part3_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_1_conv2/Conv2D/ReadVariableOp¢-resblock_part3_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_2_conv1/Conv2D/ReadVariableOp¢-resblock_part3_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_2_conv2/Conv2D/ReadVariableOp¢-resblock_part3_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_3_conv1/Conv2D/ReadVariableOp¢-resblock_part3_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_3_conv2/Conv2D/ReadVariableOp¢-resblock_part3_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_4_conv1/Conv2D/ReadVariableOp¢-resblock_part3_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_4_conv2/Conv2D/ReadVariableOp¢"upsampler_1/BiasAdd/ReadVariableOp¢!upsampler_1/Conv2D/ReadVariableOp¢"upsampler_2/BiasAdd/ReadVariableOp¢!upsampler_2/Conv2D/ReadVariableOp¶
 input_conv/Conv2D/ReadVariableOpReadVariableOp)input_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 input_conv/Conv2D/ReadVariableOpÝ
input_conv/Conv2DConv2Dinputs(input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
input_conv/Conv2D­
!input_conv/BiasAdd/ReadVariableOpReadVariableOp*input_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!input_conv/BiasAdd/ReadVariableOpÍ
input_conv/BiasAddBiasAddinput_conv/Conv2D:output:0)input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
input_conv/BiasAdd«
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddings®
zero_padding2d/PadPadinput_conv/BiasAdd:output:0$zero_padding2d/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zero_padding2d/Pad¿
#downsampler_1/Conv2D/ReadVariableOpReadVariableOp,downsampler_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02%
#downsampler_1/Conv2D/ReadVariableOpü
downsampler_1/Conv2DConv2Dzero_padding2d/Pad:output:0+downsampler_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingVALID*
strides
2
downsampler_1/Conv2D¶
$downsampler_1/BiasAdd/ReadVariableOpReadVariableOp-downsampler_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$downsampler_1/BiasAdd/ReadVariableOpÙ
downsampler_1/BiasAddBiasAdddownsampler_1/Conv2D:output:0,downsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
downsampler_1/BiasAddÚ
,resblock_part1_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_1_conv1/Conv2D/ReadVariableOp
resblock_part1_1_conv1/Conv2DConv2Ddownsampler_1/BiasAdd:output:04resblock_part1_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_1_conv1/Conv2DÑ
-resblock_part1_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_1_conv1/BiasAdd/ReadVariableOpý
resblock_part1_1_conv1/BiasAddBiasAdd&resblock_part1_1_conv1/Conv2D:output:05resblock_part1_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_1_conv1/BiasAdd§
resblock_part1_1_relu1/ReluRelu'resblock_part1_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_1_relu1/ReluÚ
,resblock_part1_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_1_conv2/Conv2D/ReadVariableOp¤
resblock_part1_1_conv2/Conv2DConv2D)resblock_part1_1_relu1/Relu:activations:04resblock_part1_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_1_conv2/Conv2DÑ
-resblock_part1_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_1_conv2/BiasAdd/ReadVariableOpý
resblock_part1_1_conv2/BiasAddBiasAdd&resblock_part1_1_conv2/Conv2D:output:05resblock_part1_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_1_conv2/BiasAdd°
tf.math.multiply/MulMultf_math_multiply_mul_x'resblock_part1_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/Mul·
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2Ú
,resblock_part1_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_2_conv1/Conv2D/ReadVariableOp
resblock_part1_2_conv1/Conv2DConv2Dtf.__operators__.add/AddV2:z:04resblock_part1_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_2_conv1/Conv2DÑ
-resblock_part1_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_2_conv1/BiasAdd/ReadVariableOpý
resblock_part1_2_conv1/BiasAddBiasAdd&resblock_part1_2_conv1/Conv2D:output:05resblock_part1_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_2_conv1/BiasAdd§
resblock_part1_2_relu1/ReluRelu'resblock_part1_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_2_relu1/ReluÚ
,resblock_part1_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_2_conv2/Conv2D/ReadVariableOp¤
resblock_part1_2_conv2/Conv2DConv2D)resblock_part1_2_relu1/Relu:activations:04resblock_part1_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_2_conv2/Conv2DÑ
-resblock_part1_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_2_conv2/BiasAdd/ReadVariableOpý
resblock_part1_2_conv2/BiasAddBiasAdd&resblock_part1_2_conv2/Conv2D:output:05resblock_part1_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_2_conv2/BiasAdd¶
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x'resblock_part1_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2Ú
,resblock_part1_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_3_conv1/Conv2D/ReadVariableOp
resblock_part1_3_conv1/Conv2DConv2D tf.__operators__.add_1/AddV2:z:04resblock_part1_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_3_conv1/Conv2DÑ
-resblock_part1_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_3_conv1/BiasAdd/ReadVariableOpý
resblock_part1_3_conv1/BiasAddBiasAdd&resblock_part1_3_conv1/Conv2D:output:05resblock_part1_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_3_conv1/BiasAdd§
resblock_part1_3_relu1/ReluRelu'resblock_part1_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_3_relu1/ReluÚ
,resblock_part1_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_3_conv2/Conv2D/ReadVariableOp¤
resblock_part1_3_conv2/Conv2DConv2D)resblock_part1_3_relu1/Relu:activations:04resblock_part1_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_3_conv2/Conv2DÑ
-resblock_part1_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_3_conv2/BiasAdd/ReadVariableOpý
resblock_part1_3_conv2/BiasAddBiasAdd&resblock_part1_3_conv2/Conv2D:output:05resblock_part1_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_3_conv2/BiasAdd¶
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x'resblock_part1_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2Ú
,resblock_part1_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_4_conv1/Conv2D/ReadVariableOp
resblock_part1_4_conv1/Conv2DConv2D tf.__operators__.add_2/AddV2:z:04resblock_part1_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_4_conv1/Conv2DÑ
-resblock_part1_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_4_conv1/BiasAdd/ReadVariableOpý
resblock_part1_4_conv1/BiasAddBiasAdd&resblock_part1_4_conv1/Conv2D:output:05resblock_part1_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_4_conv1/BiasAdd§
resblock_part1_4_relu1/ReluRelu'resblock_part1_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_4_relu1/ReluÚ
,resblock_part1_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_4_conv2/Conv2D/ReadVariableOp¤
resblock_part1_4_conv2/Conv2DConv2D)resblock_part1_4_relu1/Relu:activations:04resblock_part1_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_4_conv2/Conv2DÑ
-resblock_part1_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_4_conv2/BiasAdd/ReadVariableOpý
resblock_part1_4_conv2/BiasAddBiasAdd&resblock_part1_4_conv2/Conv2D:output:05resblock_part1_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_4_conv2/BiasAdd¶
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x'resblock_part1_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2¯
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_1/Pad/paddings¹
zero_padding2d_1/PadPad tf.__operators__.add_3/AddV2:z:0&zero_padding2d_1/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zero_padding2d_1/Pad¿
#downsampler_2/Conv2D/ReadVariableOpReadVariableOp,downsampler_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02%
#downsampler_2/Conv2D/ReadVariableOpü
downsampler_2/Conv2DConv2Dzero_padding2d_1/Pad:output:0+downsampler_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingVALID*
strides
2
downsampler_2/Conv2D¶
$downsampler_2/BiasAdd/ReadVariableOpReadVariableOp-downsampler_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$downsampler_2/BiasAdd/ReadVariableOp×
downsampler_2/BiasAddBiasAdddownsampler_2/Conv2D:output:0,downsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2
downsampler_2/BiasAddÚ
,resblock_part2_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_1_conv1/Conv2D/ReadVariableOp
resblock_part2_1_conv1/Conv2DConv2Ddownsampler_2/BiasAdd:output:04resblock_part2_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_1_conv1/Conv2DÑ
-resblock_part2_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_1_conv1/BiasAdd/ReadVariableOpû
resblock_part2_1_conv1/BiasAddBiasAdd&resblock_part2_1_conv1/Conv2D:output:05resblock_part2_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_1_conv1/BiasAdd¥
resblock_part2_1_relu1/ReluRelu'resblock_part2_1_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_1_relu1/ReluÚ
,resblock_part2_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_1_conv2/Conv2D/ReadVariableOp¢
resblock_part2_1_conv2/Conv2DConv2D)resblock_part2_1_relu1/Relu:activations:04resblock_part2_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_1_conv2/Conv2DÑ
-resblock_part2_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_1_conv2/BiasAdd/ReadVariableOpû
resblock_part2_1_conv2/BiasAddBiasAdd&resblock_part2_1_conv2/Conv2D:output:05resblock_part2_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_1_conv2/BiasAdd´
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x'resblock_part2_1_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/Mul»
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0downsampler_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2Ú
,resblock_part2_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_2_conv1/Conv2D/ReadVariableOp
resblock_part2_2_conv1/Conv2DConv2D tf.__operators__.add_4/AddV2:z:04resblock_part2_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_2_conv1/Conv2DÑ
-resblock_part2_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_2_conv1/BiasAdd/ReadVariableOpû
resblock_part2_2_conv1/BiasAddBiasAdd&resblock_part2_2_conv1/Conv2D:output:05resblock_part2_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_2_conv1/BiasAdd¥
resblock_part2_2_relu1/ReluRelu'resblock_part2_2_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_2_relu1/ReluÚ
,resblock_part2_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_2_conv2/Conv2D/ReadVariableOp¢
resblock_part2_2_conv2/Conv2DConv2D)resblock_part2_2_relu1/Relu:activations:04resblock_part2_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_2_conv2/Conv2DÑ
-resblock_part2_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_2_conv2/BiasAdd/ReadVariableOpû
resblock_part2_2_conv2/BiasAddBiasAdd&resblock_part2_2_conv2/Conv2D:output:05resblock_part2_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_2_conv2/BiasAdd´
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x'resblock_part2_2_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2Ú
,resblock_part2_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_3_conv1/Conv2D/ReadVariableOp
resblock_part2_3_conv1/Conv2DConv2D tf.__operators__.add_5/AddV2:z:04resblock_part2_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_3_conv1/Conv2DÑ
-resblock_part2_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_3_conv1/BiasAdd/ReadVariableOpû
resblock_part2_3_conv1/BiasAddBiasAdd&resblock_part2_3_conv1/Conv2D:output:05resblock_part2_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_3_conv1/BiasAdd¥
resblock_part2_3_relu1/ReluRelu'resblock_part2_3_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_3_relu1/ReluÚ
,resblock_part2_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_3_conv2/Conv2D/ReadVariableOp¢
resblock_part2_3_conv2/Conv2DConv2D)resblock_part2_3_relu1/Relu:activations:04resblock_part2_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_3_conv2/Conv2DÑ
-resblock_part2_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_3_conv2/BiasAdd/ReadVariableOpû
resblock_part2_3_conv2/BiasAddBiasAdd&resblock_part2_3_conv2/Conv2D:output:05resblock_part2_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_3_conv2/BiasAdd´
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x'resblock_part2_3_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2Ú
,resblock_part2_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_4_conv1/Conv2D/ReadVariableOp
resblock_part2_4_conv1/Conv2DConv2D tf.__operators__.add_6/AddV2:z:04resblock_part2_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_4_conv1/Conv2DÑ
-resblock_part2_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_4_conv1/BiasAdd/ReadVariableOpû
resblock_part2_4_conv1/BiasAddBiasAdd&resblock_part2_4_conv1/Conv2D:output:05resblock_part2_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_4_conv1/BiasAdd¥
resblock_part2_4_relu1/ReluRelu'resblock_part2_4_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_4_relu1/ReluÚ
,resblock_part2_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_4_conv2/Conv2D/ReadVariableOp¢
resblock_part2_4_conv2/Conv2DConv2D)resblock_part2_4_relu1/Relu:activations:04resblock_part2_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_4_conv2/Conv2DÑ
-resblock_part2_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_4_conv2/BiasAdd/ReadVariableOpû
resblock_part2_4_conv2/BiasAddBiasAdd&resblock_part2_4_conv2/Conv2D:output:05resblock_part2_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_4_conv2/BiasAdd´
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x'resblock_part2_4_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2Ú
,resblock_part2_5_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_5_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_5_conv1/Conv2D/ReadVariableOp
resblock_part2_5_conv1/Conv2DConv2D tf.__operators__.add_7/AddV2:z:04resblock_part2_5_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_5_conv1/Conv2DÑ
-resblock_part2_5_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_5_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_5_conv1/BiasAdd/ReadVariableOpû
resblock_part2_5_conv1/BiasAddBiasAdd&resblock_part2_5_conv1/Conv2D:output:05resblock_part2_5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_5_conv1/BiasAdd¥
resblock_part2_5_relu1/ReluRelu'resblock_part2_5_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_5_relu1/ReluÚ
,resblock_part2_5_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_5_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_5_conv2/Conv2D/ReadVariableOp¢
resblock_part2_5_conv2/Conv2DConv2D)resblock_part2_5_relu1/Relu:activations:04resblock_part2_5_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_5_conv2/Conv2DÑ
-resblock_part2_5_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_5_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_5_conv2/BiasAdd/ReadVariableOpû
resblock_part2_5_conv2/BiasAddBiasAdd&resblock_part2_5_conv2/Conv2D:output:05resblock_part2_5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_5_conv2/BiasAdd´
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x'resblock_part2_5_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2Ú
,resblock_part2_6_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_6_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_6_conv1/Conv2D/ReadVariableOp
resblock_part2_6_conv1/Conv2DConv2D tf.__operators__.add_8/AddV2:z:04resblock_part2_6_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_6_conv1/Conv2DÑ
-resblock_part2_6_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_6_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_6_conv1/BiasAdd/ReadVariableOpû
resblock_part2_6_conv1/BiasAddBiasAdd&resblock_part2_6_conv1/Conv2D:output:05resblock_part2_6_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_6_conv1/BiasAdd¥
resblock_part2_6_relu1/ReluRelu'resblock_part2_6_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_6_relu1/ReluÚ
,resblock_part2_6_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_6_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_6_conv2/Conv2D/ReadVariableOp¢
resblock_part2_6_conv2/Conv2DConv2D)resblock_part2_6_relu1/Relu:activations:04resblock_part2_6_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_6_conv2/Conv2DÑ
-resblock_part2_6_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_6_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_6_conv2/BiasAdd/ReadVariableOpû
resblock_part2_6_conv2/BiasAddBiasAdd&resblock_part2_6_conv2/Conv2D:output:05resblock_part2_6_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_6_conv2/BiasAdd´
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x'resblock_part2_6_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2Ú
,resblock_part2_7_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_7_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_7_conv1/Conv2D/ReadVariableOp
resblock_part2_7_conv1/Conv2DConv2D tf.__operators__.add_9/AddV2:z:04resblock_part2_7_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_7_conv1/Conv2DÑ
-resblock_part2_7_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_7_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_7_conv1/BiasAdd/ReadVariableOpû
resblock_part2_7_conv1/BiasAddBiasAdd&resblock_part2_7_conv1/Conv2D:output:05resblock_part2_7_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_7_conv1/BiasAdd¥
resblock_part2_7_relu1/ReluRelu'resblock_part2_7_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_7_relu1/ReluÚ
,resblock_part2_7_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_7_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_7_conv2/Conv2D/ReadVariableOp¢
resblock_part2_7_conv2/Conv2DConv2D)resblock_part2_7_relu1/Relu:activations:04resblock_part2_7_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_7_conv2/Conv2DÑ
-resblock_part2_7_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_7_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_7_conv2/BiasAdd/ReadVariableOpû
resblock_part2_7_conv2/BiasAddBiasAdd&resblock_part2_7_conv2/Conv2D:output:05resblock_part2_7_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_7_conv2/BiasAdd·
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x'resblock_part2_7_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2Ú
,resblock_part2_8_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_8_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_8_conv1/Conv2D/ReadVariableOp
resblock_part2_8_conv1/Conv2DConv2D!tf.__operators__.add_10/AddV2:z:04resblock_part2_8_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_8_conv1/Conv2DÑ
-resblock_part2_8_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_8_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_8_conv1/BiasAdd/ReadVariableOpû
resblock_part2_8_conv1/BiasAddBiasAdd&resblock_part2_8_conv1/Conv2D:output:05resblock_part2_8_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_8_conv1/BiasAdd¥
resblock_part2_8_relu1/ReluRelu'resblock_part2_8_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_8_relu1/ReluÚ
,resblock_part2_8_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_8_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_8_conv2/Conv2D/ReadVariableOp¢
resblock_part2_8_conv2/Conv2DConv2D)resblock_part2_8_relu1/Relu:activations:04resblock_part2_8_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_8_conv2/Conv2DÑ
-resblock_part2_8_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_8_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_8_conv2/BiasAdd/ReadVariableOpû
resblock_part2_8_conv2/BiasAddBiasAdd&resblock_part2_8_conv2/Conv2D:output:05resblock_part2_8_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_8_conv2/BiasAdd·
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x'resblock_part2_8_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2º
!upsampler_1/Conv2D/ReadVariableOpReadVariableOp*upsampler_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02#
!upsampler_1/Conv2D/ReadVariableOpú
upsampler_1/Conv2DConv2D!tf.__operators__.add_11/AddV2:z:0)upsampler_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW*
paddingSAME*
strides
2
upsampler_1/Conv2D±
"upsampler_1/BiasAdd/ReadVariableOpReadVariableOp+upsampler_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"upsampler_1/BiasAdd/ReadVariableOpÐ
upsampler_1/BiasAddBiasAddupsampler_1/Conv2D:output:0*upsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW2
upsampler_1/BiasAddÙ
!tf.nn.depth_to_space/DepthToSpaceDepthToSpaceupsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpaceÚ
,resblock_part3_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_1_conv1/Conv2D/ReadVariableOp¥
resblock_part3_1_conv1/Conv2DConv2D*tf.nn.depth_to_space/DepthToSpace:output:04resblock_part3_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_1_conv1/Conv2DÑ
-resblock_part3_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_1_conv1/BiasAdd/ReadVariableOpý
resblock_part3_1_conv1/BiasAddBiasAdd&resblock_part3_1_conv1/Conv2D:output:05resblock_part3_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_1_conv1/BiasAdd§
resblock_part3_1_relu1/ReluRelu'resblock_part3_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_1_relu1/ReluÚ
,resblock_part3_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_1_conv2/Conv2D/ReadVariableOp¤
resblock_part3_1_conv2/Conv2DConv2D)resblock_part3_1_relu1/Relu:activations:04resblock_part3_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_1_conv2/Conv2DÑ
-resblock_part3_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_1_conv2/BiasAdd/ReadVariableOpý
resblock_part3_1_conv2/BiasAddBiasAdd&resblock_part3_1_conv2/Conv2D:output:05resblock_part3_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_1_conv2/BiasAdd¹
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x'resblock_part3_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2Ú
,resblock_part3_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_2_conv1/Conv2D/ReadVariableOp
resblock_part3_2_conv1/Conv2DConv2D!tf.__operators__.add_12/AddV2:z:04resblock_part3_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_2_conv1/Conv2DÑ
-resblock_part3_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_2_conv1/BiasAdd/ReadVariableOpý
resblock_part3_2_conv1/BiasAddBiasAdd&resblock_part3_2_conv1/Conv2D:output:05resblock_part3_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_2_conv1/BiasAdd§
resblock_part3_2_relu1/ReluRelu'resblock_part3_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_2_relu1/ReluÚ
,resblock_part3_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_2_conv2/Conv2D/ReadVariableOp¤
resblock_part3_2_conv2/Conv2DConv2D)resblock_part3_2_relu1/Relu:activations:04resblock_part3_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_2_conv2/Conv2DÑ
-resblock_part3_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_2_conv2/BiasAdd/ReadVariableOpý
resblock_part3_2_conv2/BiasAddBiasAdd&resblock_part3_2_conv2/Conv2D:output:05resblock_part3_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_2_conv2/BiasAdd¹
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x'resblock_part3_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2Ú
,resblock_part3_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_3_conv1/Conv2D/ReadVariableOp
resblock_part3_3_conv1/Conv2DConv2D!tf.__operators__.add_13/AddV2:z:04resblock_part3_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_3_conv1/Conv2DÑ
-resblock_part3_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_3_conv1/BiasAdd/ReadVariableOpý
resblock_part3_3_conv1/BiasAddBiasAdd&resblock_part3_3_conv1/Conv2D:output:05resblock_part3_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_3_conv1/BiasAdd§
resblock_part3_3_relu1/ReluRelu'resblock_part3_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_3_relu1/ReluÚ
,resblock_part3_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_3_conv2/Conv2D/ReadVariableOp¤
resblock_part3_3_conv2/Conv2DConv2D)resblock_part3_3_relu1/Relu:activations:04resblock_part3_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_3_conv2/Conv2DÑ
-resblock_part3_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_3_conv2/BiasAdd/ReadVariableOpý
resblock_part3_3_conv2/BiasAddBiasAdd&resblock_part3_3_conv2/Conv2D:output:05resblock_part3_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_3_conv2/BiasAdd¹
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x'resblock_part3_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2Ú
,resblock_part3_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_4_conv1/Conv2D/ReadVariableOp
resblock_part3_4_conv1/Conv2DConv2D!tf.__operators__.add_14/AddV2:z:04resblock_part3_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_4_conv1/Conv2DÑ
-resblock_part3_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_4_conv1/BiasAdd/ReadVariableOpý
resblock_part3_4_conv1/BiasAddBiasAdd&resblock_part3_4_conv1/Conv2D:output:05resblock_part3_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_4_conv1/BiasAdd§
resblock_part3_4_relu1/ReluRelu'resblock_part3_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_4_relu1/ReluÚ
,resblock_part3_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_4_conv2/Conv2D/ReadVariableOp¤
resblock_part3_4_conv2/Conv2DConv2D)resblock_part3_4_relu1/Relu:activations:04resblock_part3_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_4_conv2/Conv2DÑ
-resblock_part3_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_4_conv2/BiasAdd/ReadVariableOpý
resblock_part3_4_conv2/BiasAddBiasAdd&resblock_part3_4_conv2/Conv2D:output:05resblock_part3_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_4_conv2/BiasAdd¹
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x'resblock_part3_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2¶
 extra_conv/Conv2D/ReadVariableOpReadVariableOp)extra_conv_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02"
 extra_conv/Conv2D/ReadVariableOpø
extra_conv/Conv2DConv2D!tf.__operators__.add_15/AddV2:z:0(extra_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
extra_conv/Conv2D­
!extra_conv/BiasAdd/ReadVariableOpReadVariableOp*extra_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!extra_conv/BiasAdd/ReadVariableOpÍ
extra_conv/BiasAddBiasAddextra_conv/Conv2D:output:0)extra_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
extra_conv/BiasAddÀ
tf.__operators__.add_16/AddV2AddV2extra_conv/BiasAdd:output:0downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2º
!upsampler_2/Conv2D/ReadVariableOpReadVariableOp*upsampler_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02#
!upsampler_2/Conv2D/ReadVariableOpü
upsampler_2/Conv2DConv2D!tf.__operators__.add_16/AddV2:z:0)upsampler_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
upsampler_2/Conv2D±
"upsampler_2/BiasAdd/ReadVariableOpReadVariableOp+upsampler_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"upsampler_2/BiasAdd/ReadVariableOpÒ
upsampler_2/BiasAddBiasAddupsampler_2/Conv2D:output:0*upsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2
upsampler_2/BiasAddÝ
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpaceupsampler_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpace¹
!output_conv/Conv2D/ReadVariableOpReadVariableOp*output_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02#
!output_conv/Conv2D/ReadVariableOp
output_conv/Conv2DConv2D,tf.nn.depth_to_space_1/DepthToSpace:output:0)output_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
output_conv/Conv2D°
"output_conv/BiasAdd/ReadVariableOpReadVariableOp+output_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"output_conv/BiasAdd/ReadVariableOpÑ
output_conv/BiasAddBiasAddoutput_conv/Conv2D:output:0*output_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2
output_conv/BiasAddÝ
IdentityIdentityoutput_conv/BiasAdd:output:0%^downsampler_1/BiasAdd/ReadVariableOp$^downsampler_1/Conv2D/ReadVariableOp%^downsampler_2/BiasAdd/ReadVariableOp$^downsampler_2/Conv2D/ReadVariableOp"^extra_conv/BiasAdd/ReadVariableOp!^extra_conv/Conv2D/ReadVariableOp"^input_conv/BiasAdd/ReadVariableOp!^input_conv/Conv2D/ReadVariableOp#^output_conv/BiasAdd/ReadVariableOp"^output_conv/Conv2D/ReadVariableOp.^resblock_part1_1_conv1/BiasAdd/ReadVariableOp-^resblock_part1_1_conv1/Conv2D/ReadVariableOp.^resblock_part1_1_conv2/BiasAdd/ReadVariableOp-^resblock_part1_1_conv2/Conv2D/ReadVariableOp.^resblock_part1_2_conv1/BiasAdd/ReadVariableOp-^resblock_part1_2_conv1/Conv2D/ReadVariableOp.^resblock_part1_2_conv2/BiasAdd/ReadVariableOp-^resblock_part1_2_conv2/Conv2D/ReadVariableOp.^resblock_part1_3_conv1/BiasAdd/ReadVariableOp-^resblock_part1_3_conv1/Conv2D/ReadVariableOp.^resblock_part1_3_conv2/BiasAdd/ReadVariableOp-^resblock_part1_3_conv2/Conv2D/ReadVariableOp.^resblock_part1_4_conv1/BiasAdd/ReadVariableOp-^resblock_part1_4_conv1/Conv2D/ReadVariableOp.^resblock_part1_4_conv2/BiasAdd/ReadVariableOp-^resblock_part1_4_conv2/Conv2D/ReadVariableOp.^resblock_part2_1_conv1/BiasAdd/ReadVariableOp-^resblock_part2_1_conv1/Conv2D/ReadVariableOp.^resblock_part2_1_conv2/BiasAdd/ReadVariableOp-^resblock_part2_1_conv2/Conv2D/ReadVariableOp.^resblock_part2_2_conv1/BiasAdd/ReadVariableOp-^resblock_part2_2_conv1/Conv2D/ReadVariableOp.^resblock_part2_2_conv2/BiasAdd/ReadVariableOp-^resblock_part2_2_conv2/Conv2D/ReadVariableOp.^resblock_part2_3_conv1/BiasAdd/ReadVariableOp-^resblock_part2_3_conv1/Conv2D/ReadVariableOp.^resblock_part2_3_conv2/BiasAdd/ReadVariableOp-^resblock_part2_3_conv2/Conv2D/ReadVariableOp.^resblock_part2_4_conv1/BiasAdd/ReadVariableOp-^resblock_part2_4_conv1/Conv2D/ReadVariableOp.^resblock_part2_4_conv2/BiasAdd/ReadVariableOp-^resblock_part2_4_conv2/Conv2D/ReadVariableOp.^resblock_part2_5_conv1/BiasAdd/ReadVariableOp-^resblock_part2_5_conv1/Conv2D/ReadVariableOp.^resblock_part2_5_conv2/BiasAdd/ReadVariableOp-^resblock_part2_5_conv2/Conv2D/ReadVariableOp.^resblock_part2_6_conv1/BiasAdd/ReadVariableOp-^resblock_part2_6_conv1/Conv2D/ReadVariableOp.^resblock_part2_6_conv2/BiasAdd/ReadVariableOp-^resblock_part2_6_conv2/Conv2D/ReadVariableOp.^resblock_part2_7_conv1/BiasAdd/ReadVariableOp-^resblock_part2_7_conv1/Conv2D/ReadVariableOp.^resblock_part2_7_conv2/BiasAdd/ReadVariableOp-^resblock_part2_7_conv2/Conv2D/ReadVariableOp.^resblock_part2_8_conv1/BiasAdd/ReadVariableOp-^resblock_part2_8_conv1/Conv2D/ReadVariableOp.^resblock_part2_8_conv2/BiasAdd/ReadVariableOp-^resblock_part2_8_conv2/Conv2D/ReadVariableOp.^resblock_part3_1_conv1/BiasAdd/ReadVariableOp-^resblock_part3_1_conv1/Conv2D/ReadVariableOp.^resblock_part3_1_conv2/BiasAdd/ReadVariableOp-^resblock_part3_1_conv2/Conv2D/ReadVariableOp.^resblock_part3_2_conv1/BiasAdd/ReadVariableOp-^resblock_part3_2_conv1/Conv2D/ReadVariableOp.^resblock_part3_2_conv2/BiasAdd/ReadVariableOp-^resblock_part3_2_conv2/Conv2D/ReadVariableOp.^resblock_part3_3_conv1/BiasAdd/ReadVariableOp-^resblock_part3_3_conv1/Conv2D/ReadVariableOp.^resblock_part3_3_conv2/BiasAdd/ReadVariableOp-^resblock_part3_3_conv2/Conv2D/ReadVariableOp.^resblock_part3_4_conv1/BiasAdd/ReadVariableOp-^resblock_part3_4_conv1/Conv2D/ReadVariableOp.^resblock_part3_4_conv2/BiasAdd/ReadVariableOp-^resblock_part3_4_conv2/Conv2D/ReadVariableOp#^upsampler_1/BiasAdd/ReadVariableOp"^upsampler_1/Conv2D/ReadVariableOp#^upsampler_2/BiasAdd/ReadVariableOp"^upsampler_2/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2L
$downsampler_1/BiasAdd/ReadVariableOp$downsampler_1/BiasAdd/ReadVariableOp2J
#downsampler_1/Conv2D/ReadVariableOp#downsampler_1/Conv2D/ReadVariableOp2L
$downsampler_2/BiasAdd/ReadVariableOp$downsampler_2/BiasAdd/ReadVariableOp2J
#downsampler_2/Conv2D/ReadVariableOp#downsampler_2/Conv2D/ReadVariableOp2F
!extra_conv/BiasAdd/ReadVariableOp!extra_conv/BiasAdd/ReadVariableOp2D
 extra_conv/Conv2D/ReadVariableOp extra_conv/Conv2D/ReadVariableOp2F
!input_conv/BiasAdd/ReadVariableOp!input_conv/BiasAdd/ReadVariableOp2D
 input_conv/Conv2D/ReadVariableOp input_conv/Conv2D/ReadVariableOp2H
"output_conv/BiasAdd/ReadVariableOp"output_conv/BiasAdd/ReadVariableOp2F
!output_conv/Conv2D/ReadVariableOp!output_conv/Conv2D/ReadVariableOp2^
-resblock_part1_1_conv1/BiasAdd/ReadVariableOp-resblock_part1_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_1_conv1/Conv2D/ReadVariableOp,resblock_part1_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_1_conv2/BiasAdd/ReadVariableOp-resblock_part1_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_1_conv2/Conv2D/ReadVariableOp,resblock_part1_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_2_conv1/BiasAdd/ReadVariableOp-resblock_part1_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_2_conv1/Conv2D/ReadVariableOp,resblock_part1_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_2_conv2/BiasAdd/ReadVariableOp-resblock_part1_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_2_conv2/Conv2D/ReadVariableOp,resblock_part1_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_3_conv1/BiasAdd/ReadVariableOp-resblock_part1_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_3_conv1/Conv2D/ReadVariableOp,resblock_part1_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_3_conv2/BiasAdd/ReadVariableOp-resblock_part1_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_3_conv2/Conv2D/ReadVariableOp,resblock_part1_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_4_conv1/BiasAdd/ReadVariableOp-resblock_part1_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_4_conv1/Conv2D/ReadVariableOp,resblock_part1_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_4_conv2/BiasAdd/ReadVariableOp-resblock_part1_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_4_conv2/Conv2D/ReadVariableOp,resblock_part1_4_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_1_conv1/BiasAdd/ReadVariableOp-resblock_part2_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_1_conv1/Conv2D/ReadVariableOp,resblock_part2_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_1_conv2/BiasAdd/ReadVariableOp-resblock_part2_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_1_conv2/Conv2D/ReadVariableOp,resblock_part2_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_2_conv1/BiasAdd/ReadVariableOp-resblock_part2_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_2_conv1/Conv2D/ReadVariableOp,resblock_part2_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_2_conv2/BiasAdd/ReadVariableOp-resblock_part2_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_2_conv2/Conv2D/ReadVariableOp,resblock_part2_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_3_conv1/BiasAdd/ReadVariableOp-resblock_part2_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_3_conv1/Conv2D/ReadVariableOp,resblock_part2_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_3_conv2/BiasAdd/ReadVariableOp-resblock_part2_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_3_conv2/Conv2D/ReadVariableOp,resblock_part2_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_4_conv1/BiasAdd/ReadVariableOp-resblock_part2_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_4_conv1/Conv2D/ReadVariableOp,resblock_part2_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_4_conv2/BiasAdd/ReadVariableOp-resblock_part2_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_4_conv2/Conv2D/ReadVariableOp,resblock_part2_4_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_5_conv1/BiasAdd/ReadVariableOp-resblock_part2_5_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_5_conv1/Conv2D/ReadVariableOp,resblock_part2_5_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_5_conv2/BiasAdd/ReadVariableOp-resblock_part2_5_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_5_conv2/Conv2D/ReadVariableOp,resblock_part2_5_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_6_conv1/BiasAdd/ReadVariableOp-resblock_part2_6_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_6_conv1/Conv2D/ReadVariableOp,resblock_part2_6_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_6_conv2/BiasAdd/ReadVariableOp-resblock_part2_6_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_6_conv2/Conv2D/ReadVariableOp,resblock_part2_6_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_7_conv1/BiasAdd/ReadVariableOp-resblock_part2_7_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_7_conv1/Conv2D/ReadVariableOp,resblock_part2_7_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_7_conv2/BiasAdd/ReadVariableOp-resblock_part2_7_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_7_conv2/Conv2D/ReadVariableOp,resblock_part2_7_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_8_conv1/BiasAdd/ReadVariableOp-resblock_part2_8_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_8_conv1/Conv2D/ReadVariableOp,resblock_part2_8_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_8_conv2/BiasAdd/ReadVariableOp-resblock_part2_8_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_8_conv2/Conv2D/ReadVariableOp,resblock_part2_8_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_1_conv1/BiasAdd/ReadVariableOp-resblock_part3_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_1_conv1/Conv2D/ReadVariableOp,resblock_part3_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_1_conv2/BiasAdd/ReadVariableOp-resblock_part3_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_1_conv2/Conv2D/ReadVariableOp,resblock_part3_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_2_conv1/BiasAdd/ReadVariableOp-resblock_part3_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_2_conv1/Conv2D/ReadVariableOp,resblock_part3_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_2_conv2/BiasAdd/ReadVariableOp-resblock_part3_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_2_conv2/Conv2D/ReadVariableOp,resblock_part3_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_3_conv1/BiasAdd/ReadVariableOp-resblock_part3_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_3_conv1/Conv2D/ReadVariableOp,resblock_part3_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_3_conv2/BiasAdd/ReadVariableOp-resblock_part3_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_3_conv2/Conv2D/ReadVariableOp,resblock_part3_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_4_conv1/BiasAdd/ReadVariableOp-resblock_part3_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_4_conv1/Conv2D/ReadVariableOp,resblock_part3_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_4_conv2/BiasAdd/ReadVariableOp-resblock_part3_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_4_conv2/Conv2D/ReadVariableOp,resblock_part3_4_conv2/Conv2D/ReadVariableOp2H
"upsampler_1/BiasAdd/ReadVariableOp"upsampler_1/BiasAdd/ReadVariableOp2F
!upsampler_1/Conv2D/ReadVariableOp!upsampler_1/Conv2D/ReadVariableOp2H
"upsampler_2/BiasAdd/ReadVariableOp"upsampler_2/BiasAdd/ReadVariableOp2F
!upsampler_2/Conv2D/ReadVariableOp!upsampler_2/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 


5__inference_resblock_part2_8_conv1_layer_call_fn_6358

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_29262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_2471

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_6219

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
©é
ò%
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_3364
input_layer
input_conv_2109
input_conv_2111
downsampler_1_2136
downsampler_1_2138
resblock_part1_1_conv1_2162
resblock_part1_1_conv1_2164
resblock_part1_1_conv2_2201
resblock_part1_1_conv2_2203
tf_math_multiply_mul_x
resblock_part1_2_conv1_2230
resblock_part1_2_conv1_2232
resblock_part1_2_conv2_2269
resblock_part1_2_conv2_2271
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_2298
resblock_part1_3_conv1_2300
resblock_part1_3_conv2_2337
resblock_part1_3_conv2_2339
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_2366
resblock_part1_4_conv1_2368
resblock_part1_4_conv2_2405
resblock_part1_4_conv2_2407
tf_math_multiply_3_mul_x
downsampler_2_2435
downsampler_2_2437
resblock_part2_1_conv1_2461
resblock_part2_1_conv1_2463
resblock_part2_1_conv2_2500
resblock_part2_1_conv2_2502
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_2529
resblock_part2_2_conv1_2531
resblock_part2_2_conv2_2568
resblock_part2_2_conv2_2570
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_2597
resblock_part2_3_conv1_2599
resblock_part2_3_conv2_2636
resblock_part2_3_conv2_2638
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_2665
resblock_part2_4_conv1_2667
resblock_part2_4_conv2_2704
resblock_part2_4_conv2_2706
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_2733
resblock_part2_5_conv1_2735
resblock_part2_5_conv2_2772
resblock_part2_5_conv2_2774
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_2801
resblock_part2_6_conv1_2803
resblock_part2_6_conv2_2840
resblock_part2_6_conv2_2842
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_2869
resblock_part2_7_conv1_2871
resblock_part2_7_conv2_2908
resblock_part2_7_conv2_2910
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_2937
resblock_part2_8_conv1_2939
resblock_part2_8_conv2_2976
resblock_part2_8_conv2_2978
tf_math_multiply_11_mul_x
upsampler_1_3005
upsampler_1_3007
resblock_part3_1_conv1_3032
resblock_part3_1_conv1_3034
resblock_part3_1_conv2_3071
resblock_part3_1_conv2_3073
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_3100
resblock_part3_2_conv1_3102
resblock_part3_2_conv2_3139
resblock_part3_2_conv2_3141
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_3168
resblock_part3_3_conv1_3170
resblock_part3_3_conv2_3207
resblock_part3_3_conv2_3209
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_3236
resblock_part3_4_conv1_3238
resblock_part3_4_conv2_3275
resblock_part3_4_conv2_3277
tf_math_multiply_15_mul_x
extra_conv_3304
extra_conv_3306
upsampler_2_3331
upsampler_2_3333
output_conv_3358
output_conv_3360
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCallª
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinput_layerinput_conv_2109input_conv_2111*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_input_conv_layer_call_and_return_conditional_losses_20982$
"input_conv/StatefulPartitionedCall
zero_padding2d/PartitionedCallPartitionedCall+input_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_20652 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_2136downsampler_1_2138*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_1_layer_call_and_return_conditional_losses_21252'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_2162resblock_part1_1_conv1_2164*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_215120
.resblock_part1_1_conv1/StatefulPartitionedCallº
&resblock_part1_1_relu1/PartitionedCallPartitionedCall7resblock_part1_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_21722(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_2201resblock_part1_1_conv2_2203*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_219020
.resblock_part1_1_conv2/StatefulPartitionedCallÀ
tf.math.multiply/MulMultf_math_multiply_mul_x7resblock_part1_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/MulÇ
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2ù
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_2230resblock_part1_2_conv1_2232*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_221920
.resblock_part1_2_conv1/StatefulPartitionedCallº
&resblock_part1_2_relu1/PartitionedCallPartitionedCall7resblock_part1_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_22402(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_2269resblock_part1_2_conv2_2271*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_225820
.resblock_part1_2_conv2/StatefulPartitionedCallÆ
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x7resblock_part1_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2û
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_2298resblock_part1_3_conv1_2300*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_228720
.resblock_part1_3_conv1/StatefulPartitionedCallº
&resblock_part1_3_relu1/PartitionedCallPartitionedCall7resblock_part1_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_23082(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_2337resblock_part1_3_conv2_2339*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_232620
.resblock_part1_3_conv2/StatefulPartitionedCallÆ
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x7resblock_part1_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2û
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_2366resblock_part1_4_conv1_2368*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_235520
.resblock_part1_4_conv1/StatefulPartitionedCallº
&resblock_part1_4_relu1/PartitionedCallPartitionedCall7resblock_part1_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_23762(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_2405resblock_part1_4_conv2_2407*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_239420
.resblock_part1_4_conv2/StatefulPartitionedCallÆ
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x7resblock_part1_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2
 zero_padding2d_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_20782"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_2435downsampler_2_2437*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_2_layer_call_and_return_conditional_losses_24242'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_2461resblock_part2_1_conv1_2463*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_245020
.resblock_part2_1_conv1/StatefulPartitionedCall¸
&resblock_part2_1_relu1/PartitionedCallPartitionedCall7resblock_part2_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_24712(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_2500resblock_part2_1_conv2_2502*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_248920
.resblock_part2_1_conv2/StatefulPartitionedCallÄ
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x7resblock_part2_1_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/MulË
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0.downsampler_2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2ù
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_2529resblock_part2_2_conv1_2531*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_251820
.resblock_part2_2_conv1/StatefulPartitionedCall¸
&resblock_part2_2_relu1/PartitionedCallPartitionedCall7resblock_part2_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_25392(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_2568resblock_part2_2_conv2_2570*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_255720
.resblock_part2_2_conv2/StatefulPartitionedCallÄ
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x7resblock_part2_2_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2ù
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_2597resblock_part2_3_conv1_2599*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_258620
.resblock_part2_3_conv1/StatefulPartitionedCall¸
&resblock_part2_3_relu1/PartitionedCallPartitionedCall7resblock_part2_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_26072(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_2636resblock_part2_3_conv2_2638*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_262520
.resblock_part2_3_conv2/StatefulPartitionedCallÄ
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x7resblock_part2_3_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2ù
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_2665resblock_part2_4_conv1_2667*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_265420
.resblock_part2_4_conv1/StatefulPartitionedCall¸
&resblock_part2_4_relu1/PartitionedCallPartitionedCall7resblock_part2_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_26752(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_2704resblock_part2_4_conv2_2706*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_269320
.resblock_part2_4_conv2/StatefulPartitionedCallÄ
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x7resblock_part2_4_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2ù
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_2733resblock_part2_5_conv1_2735*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_272220
.resblock_part2_5_conv1/StatefulPartitionedCall¸
&resblock_part2_5_relu1/PartitionedCallPartitionedCall7resblock_part2_5_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_27432(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_2772resblock_part2_5_conv2_2774*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_276120
.resblock_part2_5_conv2/StatefulPartitionedCallÄ
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x7resblock_part2_5_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2ù
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_2801resblock_part2_6_conv1_2803*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_279020
.resblock_part2_6_conv1/StatefulPartitionedCall¸
&resblock_part2_6_relu1/PartitionedCallPartitionedCall7resblock_part2_6_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_28112(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_2840resblock_part2_6_conv2_2842*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_282920
.resblock_part2_6_conv2/StatefulPartitionedCallÄ
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x7resblock_part2_6_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2ù
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_2869resblock_part2_7_conv1_2871*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_285820
.resblock_part2_7_conv1/StatefulPartitionedCall¸
&resblock_part2_7_relu1/PartitionedCallPartitionedCall7resblock_part2_7_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_28792(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_2908resblock_part2_7_conv2_2910*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_289720
.resblock_part2_7_conv2/StatefulPartitionedCallÇ
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x7resblock_part2_7_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2ú
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_2937resblock_part2_8_conv1_2939*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_292620
.resblock_part2_8_conv1/StatefulPartitionedCall¸
&resblock_part2_8_relu1/PartitionedCallPartitionedCall7resblock_part2_8_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_29472(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_2976resblock_part2_8_conv2_2978*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_296520
.resblock_part2_8_conv2/StatefulPartitionedCallÇ
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x7resblock_part2_8_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2Ä
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_3005upsampler_1_3007*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_1_layer_call_and_return_conditional_losses_29942%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_3032resblock_part3_1_conv1_3034*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_302120
.resblock_part3_1_conv1/StatefulPartitionedCallº
&resblock_part3_1_relu1/PartitionedCallPartitionedCall7resblock_part3_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_30422(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_3071resblock_part3_1_conv2_3073*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_306020
.resblock_part3_1_conv2/StatefulPartitionedCallÉ
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x7resblock_part3_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2ü
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_3100resblock_part3_2_conv1_3102*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_308920
.resblock_part3_2_conv1/StatefulPartitionedCallº
&resblock_part3_2_relu1/PartitionedCallPartitionedCall7resblock_part3_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_31102(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_3139resblock_part3_2_conv2_3141*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_312820
.resblock_part3_2_conv2/StatefulPartitionedCallÉ
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x7resblock_part3_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2ü
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_3168resblock_part3_3_conv1_3170*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_315720
.resblock_part3_3_conv1/StatefulPartitionedCallº
&resblock_part3_3_relu1/PartitionedCallPartitionedCall7resblock_part3_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_31782(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_3207resblock_part3_3_conv2_3209*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_319620
.resblock_part3_3_conv2/StatefulPartitionedCallÉ
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x7resblock_part3_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2ü
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_3236resblock_part3_4_conv1_3238*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_322520
.resblock_part3_4_conv1/StatefulPartitionedCallº
&resblock_part3_4_relu1/PartitionedCallPartitionedCall7resblock_part3_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_32462(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_3275resblock_part3_4_conv2_3277*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_326420
.resblock_part3_4_conv2/StatefulPartitionedCallÉ
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x7resblock_part3_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2À
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_3304extra_conv_3306*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_extra_conv_layer_call_and_return_conditional_losses_32932$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_3331upsampler_2_3333*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_2_layer_call_and_return_conditional_losses_33202%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_3358output_conv_3360*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_conv_layer_call_and_return_conditional_losses_33472%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
%downsampler_1/StatefulPartitionedCall%downsampler_1/StatefulPartitionedCall2N
%downsampler_2/StatefulPartitionedCall%downsampler_2/StatefulPartitionedCall2H
"extra_conv/StatefulPartitionedCall"extra_conv/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2`
.resblock_part1_1_conv1/StatefulPartitionedCall.resblock_part1_1_conv1/StatefulPartitionedCall2`
.resblock_part1_1_conv2/StatefulPartitionedCall.resblock_part1_1_conv2/StatefulPartitionedCall2`
.resblock_part1_2_conv1/StatefulPartitionedCall.resblock_part1_2_conv1/StatefulPartitionedCall2`
.resblock_part1_2_conv2/StatefulPartitionedCall.resblock_part1_2_conv2/StatefulPartitionedCall2`
.resblock_part1_3_conv1/StatefulPartitionedCall.resblock_part1_3_conv1/StatefulPartitionedCall2`
.resblock_part1_3_conv2/StatefulPartitionedCall.resblock_part1_3_conv2/StatefulPartitionedCall2`
.resblock_part1_4_conv1/StatefulPartitionedCall.resblock_part1_4_conv1/StatefulPartitionedCall2`
.resblock_part1_4_conv2/StatefulPartitionedCall.resblock_part1_4_conv2/StatefulPartitionedCall2`
.resblock_part2_1_conv1/StatefulPartitionedCall.resblock_part2_1_conv1/StatefulPartitionedCall2`
.resblock_part2_1_conv2/StatefulPartitionedCall.resblock_part2_1_conv2/StatefulPartitionedCall2`
.resblock_part2_2_conv1/StatefulPartitionedCall.resblock_part2_2_conv1/StatefulPartitionedCall2`
.resblock_part2_2_conv2/StatefulPartitionedCall.resblock_part2_2_conv2/StatefulPartitionedCall2`
.resblock_part2_3_conv1/StatefulPartitionedCall.resblock_part2_3_conv1/StatefulPartitionedCall2`
.resblock_part2_3_conv2/StatefulPartitionedCall.resblock_part2_3_conv2/StatefulPartitionedCall2`
.resblock_part2_4_conv1/StatefulPartitionedCall.resblock_part2_4_conv1/StatefulPartitionedCall2`
.resblock_part2_4_conv2/StatefulPartitionedCall.resblock_part2_4_conv2/StatefulPartitionedCall2`
.resblock_part2_5_conv1/StatefulPartitionedCall.resblock_part2_5_conv1/StatefulPartitionedCall2`
.resblock_part2_5_conv2/StatefulPartitionedCall.resblock_part2_5_conv2/StatefulPartitionedCall2`
.resblock_part2_6_conv1/StatefulPartitionedCall.resblock_part2_6_conv1/StatefulPartitionedCall2`
.resblock_part2_6_conv2/StatefulPartitionedCall.resblock_part2_6_conv2/StatefulPartitionedCall2`
.resblock_part2_7_conv1/StatefulPartitionedCall.resblock_part2_7_conv1/StatefulPartitionedCall2`
.resblock_part2_7_conv2/StatefulPartitionedCall.resblock_part2_7_conv2/StatefulPartitionedCall2`
.resblock_part2_8_conv1/StatefulPartitionedCall.resblock_part2_8_conv1/StatefulPartitionedCall2`
.resblock_part2_8_conv2/StatefulPartitionedCall.resblock_part2_8_conv2/StatefulPartitionedCall2`
.resblock_part3_1_conv1/StatefulPartitionedCall.resblock_part3_1_conv1/StatefulPartitionedCall2`
.resblock_part3_1_conv2/StatefulPartitionedCall.resblock_part3_1_conv2/StatefulPartitionedCall2`
.resblock_part3_2_conv1/StatefulPartitionedCall.resblock_part3_2_conv1/StatefulPartitionedCall2`
.resblock_part3_2_conv2/StatefulPartitionedCall.resblock_part3_2_conv2/StatefulPartitionedCall2`
.resblock_part3_3_conv1/StatefulPartitionedCall.resblock_part3_3_conv1/StatefulPartitionedCall2`
.resblock_part3_3_conv2/StatefulPartitionedCall.resblock_part3_3_conv2/StatefulPartitionedCall2`
.resblock_part3_4_conv1/StatefulPartitionedCall.resblock_part3_4_conv1/StatefulPartitionedCall2`
.resblock_part3_4_conv2/StatefulPartitionedCall.resblock_part3_4_conv2/StatefulPartitionedCall2J
#upsampler_1/StatefulPartitionedCall#upsampler_1/StatefulPartitionedCall2J
#upsampler_2/StatefulPartitionedCall#upsampler_2/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 


Þ
E__inference_upsampler_1_layer_call_and_return_conditional_losses_2994

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp»
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp 
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_3_conv2_layer_call_fn_6550

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_31962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_6253

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_4_conv1_layer_call_fn_6569

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_32252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_6378

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_6493

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_6464

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_6445

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ô
·C
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5059

inputs-
)input_conv_conv2d_readvariableop_resource.
*input_conv_biasadd_readvariableop_resource0
,downsampler_1_conv2d_readvariableop_resource1
-downsampler_1_biasadd_readvariableop_resource9
5resblock_part1_1_conv1_conv2d_readvariableop_resource:
6resblock_part1_1_conv1_biasadd_readvariableop_resource9
5resblock_part1_1_conv2_conv2d_readvariableop_resource:
6resblock_part1_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_mul_x9
5resblock_part1_2_conv1_conv2d_readvariableop_resource:
6resblock_part1_2_conv1_biasadd_readvariableop_resource9
5resblock_part1_2_conv2_conv2d_readvariableop_resource:
6resblock_part1_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_1_mul_x9
5resblock_part1_3_conv1_conv2d_readvariableop_resource:
6resblock_part1_3_conv1_biasadd_readvariableop_resource9
5resblock_part1_3_conv2_conv2d_readvariableop_resource:
6resblock_part1_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_2_mul_x9
5resblock_part1_4_conv1_conv2d_readvariableop_resource:
6resblock_part1_4_conv1_biasadd_readvariableop_resource9
5resblock_part1_4_conv2_conv2d_readvariableop_resource:
6resblock_part1_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_3_mul_x0
,downsampler_2_conv2d_readvariableop_resource1
-downsampler_2_biasadd_readvariableop_resource9
5resblock_part2_1_conv1_conv2d_readvariableop_resource:
6resblock_part2_1_conv1_biasadd_readvariableop_resource9
5resblock_part2_1_conv2_conv2d_readvariableop_resource:
6resblock_part2_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_4_mul_x9
5resblock_part2_2_conv1_conv2d_readvariableop_resource:
6resblock_part2_2_conv1_biasadd_readvariableop_resource9
5resblock_part2_2_conv2_conv2d_readvariableop_resource:
6resblock_part2_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_5_mul_x9
5resblock_part2_3_conv1_conv2d_readvariableop_resource:
6resblock_part2_3_conv1_biasadd_readvariableop_resource9
5resblock_part2_3_conv2_conv2d_readvariableop_resource:
6resblock_part2_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_6_mul_x9
5resblock_part2_4_conv1_conv2d_readvariableop_resource:
6resblock_part2_4_conv1_biasadd_readvariableop_resource9
5resblock_part2_4_conv2_conv2d_readvariableop_resource:
6resblock_part2_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_7_mul_x9
5resblock_part2_5_conv1_conv2d_readvariableop_resource:
6resblock_part2_5_conv1_biasadd_readvariableop_resource9
5resblock_part2_5_conv2_conv2d_readvariableop_resource:
6resblock_part2_5_conv2_biasadd_readvariableop_resource
tf_math_multiply_8_mul_x9
5resblock_part2_6_conv1_conv2d_readvariableop_resource:
6resblock_part2_6_conv1_biasadd_readvariableop_resource9
5resblock_part2_6_conv2_conv2d_readvariableop_resource:
6resblock_part2_6_conv2_biasadd_readvariableop_resource
tf_math_multiply_9_mul_x9
5resblock_part2_7_conv1_conv2d_readvariableop_resource:
6resblock_part2_7_conv1_biasadd_readvariableop_resource9
5resblock_part2_7_conv2_conv2d_readvariableop_resource:
6resblock_part2_7_conv2_biasadd_readvariableop_resource
tf_math_multiply_10_mul_x9
5resblock_part2_8_conv1_conv2d_readvariableop_resource:
6resblock_part2_8_conv1_biasadd_readvariableop_resource9
5resblock_part2_8_conv2_conv2d_readvariableop_resource:
6resblock_part2_8_conv2_biasadd_readvariableop_resource
tf_math_multiply_11_mul_x.
*upsampler_1_conv2d_readvariableop_resource/
+upsampler_1_biasadd_readvariableop_resource9
5resblock_part3_1_conv1_conv2d_readvariableop_resource:
6resblock_part3_1_conv1_biasadd_readvariableop_resource9
5resblock_part3_1_conv2_conv2d_readvariableop_resource:
6resblock_part3_1_conv2_biasadd_readvariableop_resource
tf_math_multiply_12_mul_x9
5resblock_part3_2_conv1_conv2d_readvariableop_resource:
6resblock_part3_2_conv1_biasadd_readvariableop_resource9
5resblock_part3_2_conv2_conv2d_readvariableop_resource:
6resblock_part3_2_conv2_biasadd_readvariableop_resource
tf_math_multiply_13_mul_x9
5resblock_part3_3_conv1_conv2d_readvariableop_resource:
6resblock_part3_3_conv1_biasadd_readvariableop_resource9
5resblock_part3_3_conv2_conv2d_readvariableop_resource:
6resblock_part3_3_conv2_biasadd_readvariableop_resource
tf_math_multiply_14_mul_x9
5resblock_part3_4_conv1_conv2d_readvariableop_resource:
6resblock_part3_4_conv1_biasadd_readvariableop_resource9
5resblock_part3_4_conv2_conv2d_readvariableop_resource:
6resblock_part3_4_conv2_biasadd_readvariableop_resource
tf_math_multiply_15_mul_x-
)extra_conv_conv2d_readvariableop_resource.
*extra_conv_biasadd_readvariableop_resource.
*upsampler_2_conv2d_readvariableop_resource/
+upsampler_2_biasadd_readvariableop_resource.
*output_conv_conv2d_readvariableop_resource/
+output_conv_biasadd_readvariableop_resource
identity¢$downsampler_1/BiasAdd/ReadVariableOp¢#downsampler_1/Conv2D/ReadVariableOp¢$downsampler_2/BiasAdd/ReadVariableOp¢#downsampler_2/Conv2D/ReadVariableOp¢!extra_conv/BiasAdd/ReadVariableOp¢ extra_conv/Conv2D/ReadVariableOp¢!input_conv/BiasAdd/ReadVariableOp¢ input_conv/Conv2D/ReadVariableOp¢"output_conv/BiasAdd/ReadVariableOp¢!output_conv/Conv2D/ReadVariableOp¢-resblock_part1_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_1_conv1/Conv2D/ReadVariableOp¢-resblock_part1_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_1_conv2/Conv2D/ReadVariableOp¢-resblock_part1_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_2_conv1/Conv2D/ReadVariableOp¢-resblock_part1_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_2_conv2/Conv2D/ReadVariableOp¢-resblock_part1_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_3_conv1/Conv2D/ReadVariableOp¢-resblock_part1_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_3_conv2/Conv2D/ReadVariableOp¢-resblock_part1_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part1_4_conv1/Conv2D/ReadVariableOp¢-resblock_part1_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part1_4_conv2/Conv2D/ReadVariableOp¢-resblock_part2_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_1_conv1/Conv2D/ReadVariableOp¢-resblock_part2_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_1_conv2/Conv2D/ReadVariableOp¢-resblock_part2_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_2_conv1/Conv2D/ReadVariableOp¢-resblock_part2_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_2_conv2/Conv2D/ReadVariableOp¢-resblock_part2_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_3_conv1/Conv2D/ReadVariableOp¢-resblock_part2_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_3_conv2/Conv2D/ReadVariableOp¢-resblock_part2_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_4_conv1/Conv2D/ReadVariableOp¢-resblock_part2_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_4_conv2/Conv2D/ReadVariableOp¢-resblock_part2_5_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_5_conv1/Conv2D/ReadVariableOp¢-resblock_part2_5_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_5_conv2/Conv2D/ReadVariableOp¢-resblock_part2_6_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_6_conv1/Conv2D/ReadVariableOp¢-resblock_part2_6_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_6_conv2/Conv2D/ReadVariableOp¢-resblock_part2_7_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_7_conv1/Conv2D/ReadVariableOp¢-resblock_part2_7_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_7_conv2/Conv2D/ReadVariableOp¢-resblock_part2_8_conv1/BiasAdd/ReadVariableOp¢,resblock_part2_8_conv1/Conv2D/ReadVariableOp¢-resblock_part2_8_conv2/BiasAdd/ReadVariableOp¢,resblock_part2_8_conv2/Conv2D/ReadVariableOp¢-resblock_part3_1_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_1_conv1/Conv2D/ReadVariableOp¢-resblock_part3_1_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_1_conv2/Conv2D/ReadVariableOp¢-resblock_part3_2_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_2_conv1/Conv2D/ReadVariableOp¢-resblock_part3_2_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_2_conv2/Conv2D/ReadVariableOp¢-resblock_part3_3_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_3_conv1/Conv2D/ReadVariableOp¢-resblock_part3_3_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_3_conv2/Conv2D/ReadVariableOp¢-resblock_part3_4_conv1/BiasAdd/ReadVariableOp¢,resblock_part3_4_conv1/Conv2D/ReadVariableOp¢-resblock_part3_4_conv2/BiasAdd/ReadVariableOp¢,resblock_part3_4_conv2/Conv2D/ReadVariableOp¢"upsampler_1/BiasAdd/ReadVariableOp¢!upsampler_1/Conv2D/ReadVariableOp¢"upsampler_2/BiasAdd/ReadVariableOp¢!upsampler_2/Conv2D/ReadVariableOp¶
 input_conv/Conv2D/ReadVariableOpReadVariableOp)input_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 input_conv/Conv2D/ReadVariableOpÝ
input_conv/Conv2DConv2Dinputs(input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
input_conv/Conv2D­
!input_conv/BiasAdd/ReadVariableOpReadVariableOp*input_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!input_conv/BiasAdd/ReadVariableOpÍ
input_conv/BiasAddBiasAddinput_conv/Conv2D:output:0)input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
input_conv/BiasAdd«
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddings®
zero_padding2d/PadPadinput_conv/BiasAdd:output:0$zero_padding2d/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zero_padding2d/Pad¿
#downsampler_1/Conv2D/ReadVariableOpReadVariableOp,downsampler_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02%
#downsampler_1/Conv2D/ReadVariableOpü
downsampler_1/Conv2DConv2Dzero_padding2d/Pad:output:0+downsampler_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingVALID*
strides
2
downsampler_1/Conv2D¶
$downsampler_1/BiasAdd/ReadVariableOpReadVariableOp-downsampler_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$downsampler_1/BiasAdd/ReadVariableOpÙ
downsampler_1/BiasAddBiasAdddownsampler_1/Conv2D:output:0,downsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
downsampler_1/BiasAddÚ
,resblock_part1_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_1_conv1/Conv2D/ReadVariableOp
resblock_part1_1_conv1/Conv2DConv2Ddownsampler_1/BiasAdd:output:04resblock_part1_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_1_conv1/Conv2DÑ
-resblock_part1_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_1_conv1/BiasAdd/ReadVariableOpý
resblock_part1_1_conv1/BiasAddBiasAdd&resblock_part1_1_conv1/Conv2D:output:05resblock_part1_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_1_conv1/BiasAdd§
resblock_part1_1_relu1/ReluRelu'resblock_part1_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_1_relu1/ReluÚ
,resblock_part1_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_1_conv2/Conv2D/ReadVariableOp¤
resblock_part1_1_conv2/Conv2DConv2D)resblock_part1_1_relu1/Relu:activations:04resblock_part1_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_1_conv2/Conv2DÑ
-resblock_part1_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_1_conv2/BiasAdd/ReadVariableOpý
resblock_part1_1_conv2/BiasAddBiasAdd&resblock_part1_1_conv2/Conv2D:output:05resblock_part1_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_1_conv2/BiasAdd°
tf.math.multiply/MulMultf_math_multiply_mul_x'resblock_part1_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/Mul·
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2Ú
,resblock_part1_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_2_conv1/Conv2D/ReadVariableOp
resblock_part1_2_conv1/Conv2DConv2Dtf.__operators__.add/AddV2:z:04resblock_part1_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_2_conv1/Conv2DÑ
-resblock_part1_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_2_conv1/BiasAdd/ReadVariableOpý
resblock_part1_2_conv1/BiasAddBiasAdd&resblock_part1_2_conv1/Conv2D:output:05resblock_part1_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_2_conv1/BiasAdd§
resblock_part1_2_relu1/ReluRelu'resblock_part1_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_2_relu1/ReluÚ
,resblock_part1_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_2_conv2/Conv2D/ReadVariableOp¤
resblock_part1_2_conv2/Conv2DConv2D)resblock_part1_2_relu1/Relu:activations:04resblock_part1_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_2_conv2/Conv2DÑ
-resblock_part1_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_2_conv2/BiasAdd/ReadVariableOpý
resblock_part1_2_conv2/BiasAddBiasAdd&resblock_part1_2_conv2/Conv2D:output:05resblock_part1_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_2_conv2/BiasAdd¶
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x'resblock_part1_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2Ú
,resblock_part1_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_3_conv1/Conv2D/ReadVariableOp
resblock_part1_3_conv1/Conv2DConv2D tf.__operators__.add_1/AddV2:z:04resblock_part1_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_3_conv1/Conv2DÑ
-resblock_part1_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_3_conv1/BiasAdd/ReadVariableOpý
resblock_part1_3_conv1/BiasAddBiasAdd&resblock_part1_3_conv1/Conv2D:output:05resblock_part1_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_3_conv1/BiasAdd§
resblock_part1_3_relu1/ReluRelu'resblock_part1_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_3_relu1/ReluÚ
,resblock_part1_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_3_conv2/Conv2D/ReadVariableOp¤
resblock_part1_3_conv2/Conv2DConv2D)resblock_part1_3_relu1/Relu:activations:04resblock_part1_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_3_conv2/Conv2DÑ
-resblock_part1_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_3_conv2/BiasAdd/ReadVariableOpý
resblock_part1_3_conv2/BiasAddBiasAdd&resblock_part1_3_conv2/Conv2D:output:05resblock_part1_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_3_conv2/BiasAdd¶
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x'resblock_part1_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2Ú
,resblock_part1_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_4_conv1/Conv2D/ReadVariableOp
resblock_part1_4_conv1/Conv2DConv2D tf.__operators__.add_2/AddV2:z:04resblock_part1_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_4_conv1/Conv2DÑ
-resblock_part1_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_4_conv1/BiasAdd/ReadVariableOpý
resblock_part1_4_conv1/BiasAddBiasAdd&resblock_part1_4_conv1/Conv2D:output:05resblock_part1_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_4_conv1/BiasAdd§
resblock_part1_4_relu1/ReluRelu'resblock_part1_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part1_4_relu1/ReluÚ
,resblock_part1_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part1_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part1_4_conv2/Conv2D/ReadVariableOp¤
resblock_part1_4_conv2/Conv2DConv2D)resblock_part1_4_relu1/Relu:activations:04resblock_part1_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part1_4_conv2/Conv2DÑ
-resblock_part1_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part1_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part1_4_conv2/BiasAdd/ReadVariableOpý
resblock_part1_4_conv2/BiasAddBiasAdd&resblock_part1_4_conv2/Conv2D:output:05resblock_part1_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part1_4_conv2/BiasAdd¶
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x'resblock_part1_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2¯
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_1/Pad/paddings¹
zero_padding2d_1/PadPad tf.__operators__.add_3/AddV2:z:0&zero_padding2d_1/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
zero_padding2d_1/Pad¿
#downsampler_2/Conv2D/ReadVariableOpReadVariableOp,downsampler_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02%
#downsampler_2/Conv2D/ReadVariableOpü
downsampler_2/Conv2DConv2Dzero_padding2d_1/Pad:output:0+downsampler_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingVALID*
strides
2
downsampler_2/Conv2D¶
$downsampler_2/BiasAdd/ReadVariableOpReadVariableOp-downsampler_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$downsampler_2/BiasAdd/ReadVariableOp×
downsampler_2/BiasAddBiasAdddownsampler_2/Conv2D:output:0,downsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2
downsampler_2/BiasAddÚ
,resblock_part2_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_1_conv1/Conv2D/ReadVariableOp
resblock_part2_1_conv1/Conv2DConv2Ddownsampler_2/BiasAdd:output:04resblock_part2_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_1_conv1/Conv2DÑ
-resblock_part2_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_1_conv1/BiasAdd/ReadVariableOpû
resblock_part2_1_conv1/BiasAddBiasAdd&resblock_part2_1_conv1/Conv2D:output:05resblock_part2_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_1_conv1/BiasAdd¥
resblock_part2_1_relu1/ReluRelu'resblock_part2_1_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_1_relu1/ReluÚ
,resblock_part2_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_1_conv2/Conv2D/ReadVariableOp¢
resblock_part2_1_conv2/Conv2DConv2D)resblock_part2_1_relu1/Relu:activations:04resblock_part2_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_1_conv2/Conv2DÑ
-resblock_part2_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_1_conv2/BiasAdd/ReadVariableOpû
resblock_part2_1_conv2/BiasAddBiasAdd&resblock_part2_1_conv2/Conv2D:output:05resblock_part2_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_1_conv2/BiasAdd´
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x'resblock_part2_1_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/Mul»
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0downsampler_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2Ú
,resblock_part2_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_2_conv1/Conv2D/ReadVariableOp
resblock_part2_2_conv1/Conv2DConv2D tf.__operators__.add_4/AddV2:z:04resblock_part2_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_2_conv1/Conv2DÑ
-resblock_part2_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_2_conv1/BiasAdd/ReadVariableOpû
resblock_part2_2_conv1/BiasAddBiasAdd&resblock_part2_2_conv1/Conv2D:output:05resblock_part2_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_2_conv1/BiasAdd¥
resblock_part2_2_relu1/ReluRelu'resblock_part2_2_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_2_relu1/ReluÚ
,resblock_part2_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_2_conv2/Conv2D/ReadVariableOp¢
resblock_part2_2_conv2/Conv2DConv2D)resblock_part2_2_relu1/Relu:activations:04resblock_part2_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_2_conv2/Conv2DÑ
-resblock_part2_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_2_conv2/BiasAdd/ReadVariableOpû
resblock_part2_2_conv2/BiasAddBiasAdd&resblock_part2_2_conv2/Conv2D:output:05resblock_part2_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_2_conv2/BiasAdd´
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x'resblock_part2_2_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2Ú
,resblock_part2_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_3_conv1/Conv2D/ReadVariableOp
resblock_part2_3_conv1/Conv2DConv2D tf.__operators__.add_5/AddV2:z:04resblock_part2_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_3_conv1/Conv2DÑ
-resblock_part2_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_3_conv1/BiasAdd/ReadVariableOpû
resblock_part2_3_conv1/BiasAddBiasAdd&resblock_part2_3_conv1/Conv2D:output:05resblock_part2_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_3_conv1/BiasAdd¥
resblock_part2_3_relu1/ReluRelu'resblock_part2_3_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_3_relu1/ReluÚ
,resblock_part2_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_3_conv2/Conv2D/ReadVariableOp¢
resblock_part2_3_conv2/Conv2DConv2D)resblock_part2_3_relu1/Relu:activations:04resblock_part2_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_3_conv2/Conv2DÑ
-resblock_part2_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_3_conv2/BiasAdd/ReadVariableOpû
resblock_part2_3_conv2/BiasAddBiasAdd&resblock_part2_3_conv2/Conv2D:output:05resblock_part2_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_3_conv2/BiasAdd´
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x'resblock_part2_3_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2Ú
,resblock_part2_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_4_conv1/Conv2D/ReadVariableOp
resblock_part2_4_conv1/Conv2DConv2D tf.__operators__.add_6/AddV2:z:04resblock_part2_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_4_conv1/Conv2DÑ
-resblock_part2_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_4_conv1/BiasAdd/ReadVariableOpû
resblock_part2_4_conv1/BiasAddBiasAdd&resblock_part2_4_conv1/Conv2D:output:05resblock_part2_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_4_conv1/BiasAdd¥
resblock_part2_4_relu1/ReluRelu'resblock_part2_4_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_4_relu1/ReluÚ
,resblock_part2_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_4_conv2/Conv2D/ReadVariableOp¢
resblock_part2_4_conv2/Conv2DConv2D)resblock_part2_4_relu1/Relu:activations:04resblock_part2_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_4_conv2/Conv2DÑ
-resblock_part2_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_4_conv2/BiasAdd/ReadVariableOpû
resblock_part2_4_conv2/BiasAddBiasAdd&resblock_part2_4_conv2/Conv2D:output:05resblock_part2_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_4_conv2/BiasAdd´
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x'resblock_part2_4_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2Ú
,resblock_part2_5_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_5_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_5_conv1/Conv2D/ReadVariableOp
resblock_part2_5_conv1/Conv2DConv2D tf.__operators__.add_7/AddV2:z:04resblock_part2_5_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_5_conv1/Conv2DÑ
-resblock_part2_5_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_5_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_5_conv1/BiasAdd/ReadVariableOpû
resblock_part2_5_conv1/BiasAddBiasAdd&resblock_part2_5_conv1/Conv2D:output:05resblock_part2_5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_5_conv1/BiasAdd¥
resblock_part2_5_relu1/ReluRelu'resblock_part2_5_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_5_relu1/ReluÚ
,resblock_part2_5_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_5_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_5_conv2/Conv2D/ReadVariableOp¢
resblock_part2_5_conv2/Conv2DConv2D)resblock_part2_5_relu1/Relu:activations:04resblock_part2_5_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_5_conv2/Conv2DÑ
-resblock_part2_5_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_5_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_5_conv2/BiasAdd/ReadVariableOpû
resblock_part2_5_conv2/BiasAddBiasAdd&resblock_part2_5_conv2/Conv2D:output:05resblock_part2_5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_5_conv2/BiasAdd´
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x'resblock_part2_5_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2Ú
,resblock_part2_6_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_6_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_6_conv1/Conv2D/ReadVariableOp
resblock_part2_6_conv1/Conv2DConv2D tf.__operators__.add_8/AddV2:z:04resblock_part2_6_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_6_conv1/Conv2DÑ
-resblock_part2_6_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_6_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_6_conv1/BiasAdd/ReadVariableOpû
resblock_part2_6_conv1/BiasAddBiasAdd&resblock_part2_6_conv1/Conv2D:output:05resblock_part2_6_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_6_conv1/BiasAdd¥
resblock_part2_6_relu1/ReluRelu'resblock_part2_6_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_6_relu1/ReluÚ
,resblock_part2_6_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_6_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_6_conv2/Conv2D/ReadVariableOp¢
resblock_part2_6_conv2/Conv2DConv2D)resblock_part2_6_relu1/Relu:activations:04resblock_part2_6_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_6_conv2/Conv2DÑ
-resblock_part2_6_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_6_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_6_conv2/BiasAdd/ReadVariableOpû
resblock_part2_6_conv2/BiasAddBiasAdd&resblock_part2_6_conv2/Conv2D:output:05resblock_part2_6_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_6_conv2/BiasAdd´
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x'resblock_part2_6_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2Ú
,resblock_part2_7_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_7_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_7_conv1/Conv2D/ReadVariableOp
resblock_part2_7_conv1/Conv2DConv2D tf.__operators__.add_9/AddV2:z:04resblock_part2_7_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_7_conv1/Conv2DÑ
-resblock_part2_7_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_7_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_7_conv1/BiasAdd/ReadVariableOpû
resblock_part2_7_conv1/BiasAddBiasAdd&resblock_part2_7_conv1/Conv2D:output:05resblock_part2_7_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_7_conv1/BiasAdd¥
resblock_part2_7_relu1/ReluRelu'resblock_part2_7_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_7_relu1/ReluÚ
,resblock_part2_7_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_7_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_7_conv2/Conv2D/ReadVariableOp¢
resblock_part2_7_conv2/Conv2DConv2D)resblock_part2_7_relu1/Relu:activations:04resblock_part2_7_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_7_conv2/Conv2DÑ
-resblock_part2_7_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_7_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_7_conv2/BiasAdd/ReadVariableOpû
resblock_part2_7_conv2/BiasAddBiasAdd&resblock_part2_7_conv2/Conv2D:output:05resblock_part2_7_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_7_conv2/BiasAdd·
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x'resblock_part2_7_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2Ú
,resblock_part2_8_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_8_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_8_conv1/Conv2D/ReadVariableOp
resblock_part2_8_conv1/Conv2DConv2D!tf.__operators__.add_10/AddV2:z:04resblock_part2_8_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_8_conv1/Conv2DÑ
-resblock_part2_8_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_8_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_8_conv1/BiasAdd/ReadVariableOpû
resblock_part2_8_conv1/BiasAddBiasAdd&resblock_part2_8_conv1/Conv2D:output:05resblock_part2_8_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_8_conv1/BiasAdd¥
resblock_part2_8_relu1/ReluRelu'resblock_part2_8_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
resblock_part2_8_relu1/ReluÚ
,resblock_part2_8_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part2_8_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part2_8_conv2/Conv2D/ReadVariableOp¢
resblock_part2_8_conv2/Conv2DConv2D)resblock_part2_8_relu1/Relu:activations:04resblock_part2_8_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part2_8_conv2/Conv2DÑ
-resblock_part2_8_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part2_8_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part2_8_conv2/BiasAdd/ReadVariableOpû
resblock_part2_8_conv2/BiasAddBiasAdd&resblock_part2_8_conv2/Conv2D:output:05resblock_part2_8_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2 
resblock_part2_8_conv2/BiasAdd·
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x'resblock_part2_8_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2º
!upsampler_1/Conv2D/ReadVariableOpReadVariableOp*upsampler_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02#
!upsampler_1/Conv2D/ReadVariableOpú
upsampler_1/Conv2DConv2D!tf.__operators__.add_11/AddV2:z:0)upsampler_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW*
paddingSAME*
strides
2
upsampler_1/Conv2D±
"upsampler_1/BiasAdd/ReadVariableOpReadVariableOp+upsampler_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"upsampler_1/BiasAdd/ReadVariableOpÐ
upsampler_1/BiasAddBiasAddupsampler_1/Conv2D:output:0*upsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW2
upsampler_1/BiasAddÙ
!tf.nn.depth_to_space/DepthToSpaceDepthToSpaceupsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpaceÚ
,resblock_part3_1_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_1_conv1/Conv2D/ReadVariableOp¥
resblock_part3_1_conv1/Conv2DConv2D*tf.nn.depth_to_space/DepthToSpace:output:04resblock_part3_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_1_conv1/Conv2DÑ
-resblock_part3_1_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_1_conv1/BiasAdd/ReadVariableOpý
resblock_part3_1_conv1/BiasAddBiasAdd&resblock_part3_1_conv1/Conv2D:output:05resblock_part3_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_1_conv1/BiasAdd§
resblock_part3_1_relu1/ReluRelu'resblock_part3_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_1_relu1/ReluÚ
,resblock_part3_1_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_1_conv2/Conv2D/ReadVariableOp¤
resblock_part3_1_conv2/Conv2DConv2D)resblock_part3_1_relu1/Relu:activations:04resblock_part3_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_1_conv2/Conv2DÑ
-resblock_part3_1_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_1_conv2/BiasAdd/ReadVariableOpý
resblock_part3_1_conv2/BiasAddBiasAdd&resblock_part3_1_conv2/Conv2D:output:05resblock_part3_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_1_conv2/BiasAdd¹
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x'resblock_part3_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2Ú
,resblock_part3_2_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_2_conv1/Conv2D/ReadVariableOp
resblock_part3_2_conv1/Conv2DConv2D!tf.__operators__.add_12/AddV2:z:04resblock_part3_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_2_conv1/Conv2DÑ
-resblock_part3_2_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_2_conv1/BiasAdd/ReadVariableOpý
resblock_part3_2_conv1/BiasAddBiasAdd&resblock_part3_2_conv1/Conv2D:output:05resblock_part3_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_2_conv1/BiasAdd§
resblock_part3_2_relu1/ReluRelu'resblock_part3_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_2_relu1/ReluÚ
,resblock_part3_2_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_2_conv2/Conv2D/ReadVariableOp¤
resblock_part3_2_conv2/Conv2DConv2D)resblock_part3_2_relu1/Relu:activations:04resblock_part3_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_2_conv2/Conv2DÑ
-resblock_part3_2_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_2_conv2/BiasAdd/ReadVariableOpý
resblock_part3_2_conv2/BiasAddBiasAdd&resblock_part3_2_conv2/Conv2D:output:05resblock_part3_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_2_conv2/BiasAdd¹
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x'resblock_part3_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2Ú
,resblock_part3_3_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_3_conv1/Conv2D/ReadVariableOp
resblock_part3_3_conv1/Conv2DConv2D!tf.__operators__.add_13/AddV2:z:04resblock_part3_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_3_conv1/Conv2DÑ
-resblock_part3_3_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_3_conv1/BiasAdd/ReadVariableOpý
resblock_part3_3_conv1/BiasAddBiasAdd&resblock_part3_3_conv1/Conv2D:output:05resblock_part3_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_3_conv1/BiasAdd§
resblock_part3_3_relu1/ReluRelu'resblock_part3_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_3_relu1/ReluÚ
,resblock_part3_3_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_3_conv2/Conv2D/ReadVariableOp¤
resblock_part3_3_conv2/Conv2DConv2D)resblock_part3_3_relu1/Relu:activations:04resblock_part3_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_3_conv2/Conv2DÑ
-resblock_part3_3_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_3_conv2/BiasAdd/ReadVariableOpý
resblock_part3_3_conv2/BiasAddBiasAdd&resblock_part3_3_conv2/Conv2D:output:05resblock_part3_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_3_conv2/BiasAdd¹
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x'resblock_part3_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2Ú
,resblock_part3_4_conv1/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_4_conv1/Conv2D/ReadVariableOp
resblock_part3_4_conv1/Conv2DConv2D!tf.__operators__.add_14/AddV2:z:04resblock_part3_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_4_conv1/Conv2DÑ
-resblock_part3_4_conv1/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_4_conv1/BiasAdd/ReadVariableOpý
resblock_part3_4_conv1/BiasAddBiasAdd&resblock_part3_4_conv1/Conv2D:output:05resblock_part3_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_4_conv1/BiasAdd§
resblock_part3_4_relu1/ReluRelu'resblock_part3_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
resblock_part3_4_relu1/ReluÚ
,resblock_part3_4_conv2/Conv2D/ReadVariableOpReadVariableOp5resblock_part3_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,resblock_part3_4_conv2/Conv2D/ReadVariableOp¤
resblock_part3_4_conv2/Conv2DConv2D)resblock_part3_4_relu1/Relu:activations:04resblock_part3_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
resblock_part3_4_conv2/Conv2DÑ
-resblock_part3_4_conv2/BiasAdd/ReadVariableOpReadVariableOp6resblock_part3_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-resblock_part3_4_conv2/BiasAdd/ReadVariableOpý
resblock_part3_4_conv2/BiasAddBiasAdd&resblock_part3_4_conv2/Conv2D:output:05resblock_part3_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2 
resblock_part3_4_conv2/BiasAdd¹
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x'resblock_part3_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2¶
 extra_conv/Conv2D/ReadVariableOpReadVariableOp)extra_conv_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02"
 extra_conv/Conv2D/ReadVariableOpø
extra_conv/Conv2DConv2D!tf.__operators__.add_15/AddV2:z:0(extra_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
extra_conv/Conv2D­
!extra_conv/BiasAdd/ReadVariableOpReadVariableOp*extra_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!extra_conv/BiasAdd/ReadVariableOpÍ
extra_conv/BiasAddBiasAddextra_conv/Conv2D:output:0)extra_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2
extra_conv/BiasAddÀ
tf.__operators__.add_16/AddV2AddV2extra_conv/BiasAdd:output:0downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2º
!upsampler_2/Conv2D/ReadVariableOpReadVariableOp*upsampler_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02#
!upsampler_2/Conv2D/ReadVariableOpü
upsampler_2/Conv2DConv2D!tf.__operators__.add_16/AddV2:z:0)upsampler_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
upsampler_2/Conv2D±
"upsampler_2/BiasAdd/ReadVariableOpReadVariableOp+upsampler_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"upsampler_2/BiasAdd/ReadVariableOpÒ
upsampler_2/BiasAddBiasAddupsampler_2/Conv2D:output:0*upsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2
upsampler_2/BiasAddÝ
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpaceupsampler_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpace¹
!output_conv/Conv2D/ReadVariableOpReadVariableOp*output_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02#
!output_conv/Conv2D/ReadVariableOp
output_conv/Conv2DConv2D,tf.nn.depth_to_space_1/DepthToSpace:output:0)output_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
output_conv/Conv2D°
"output_conv/BiasAdd/ReadVariableOpReadVariableOp+output_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"output_conv/BiasAdd/ReadVariableOpÑ
output_conv/BiasAddBiasAddoutput_conv/Conv2D:output:0*output_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2
output_conv/BiasAddÝ
IdentityIdentityoutput_conv/BiasAdd:output:0%^downsampler_1/BiasAdd/ReadVariableOp$^downsampler_1/Conv2D/ReadVariableOp%^downsampler_2/BiasAdd/ReadVariableOp$^downsampler_2/Conv2D/ReadVariableOp"^extra_conv/BiasAdd/ReadVariableOp!^extra_conv/Conv2D/ReadVariableOp"^input_conv/BiasAdd/ReadVariableOp!^input_conv/Conv2D/ReadVariableOp#^output_conv/BiasAdd/ReadVariableOp"^output_conv/Conv2D/ReadVariableOp.^resblock_part1_1_conv1/BiasAdd/ReadVariableOp-^resblock_part1_1_conv1/Conv2D/ReadVariableOp.^resblock_part1_1_conv2/BiasAdd/ReadVariableOp-^resblock_part1_1_conv2/Conv2D/ReadVariableOp.^resblock_part1_2_conv1/BiasAdd/ReadVariableOp-^resblock_part1_2_conv1/Conv2D/ReadVariableOp.^resblock_part1_2_conv2/BiasAdd/ReadVariableOp-^resblock_part1_2_conv2/Conv2D/ReadVariableOp.^resblock_part1_3_conv1/BiasAdd/ReadVariableOp-^resblock_part1_3_conv1/Conv2D/ReadVariableOp.^resblock_part1_3_conv2/BiasAdd/ReadVariableOp-^resblock_part1_3_conv2/Conv2D/ReadVariableOp.^resblock_part1_4_conv1/BiasAdd/ReadVariableOp-^resblock_part1_4_conv1/Conv2D/ReadVariableOp.^resblock_part1_4_conv2/BiasAdd/ReadVariableOp-^resblock_part1_4_conv2/Conv2D/ReadVariableOp.^resblock_part2_1_conv1/BiasAdd/ReadVariableOp-^resblock_part2_1_conv1/Conv2D/ReadVariableOp.^resblock_part2_1_conv2/BiasAdd/ReadVariableOp-^resblock_part2_1_conv2/Conv2D/ReadVariableOp.^resblock_part2_2_conv1/BiasAdd/ReadVariableOp-^resblock_part2_2_conv1/Conv2D/ReadVariableOp.^resblock_part2_2_conv2/BiasAdd/ReadVariableOp-^resblock_part2_2_conv2/Conv2D/ReadVariableOp.^resblock_part2_3_conv1/BiasAdd/ReadVariableOp-^resblock_part2_3_conv1/Conv2D/ReadVariableOp.^resblock_part2_3_conv2/BiasAdd/ReadVariableOp-^resblock_part2_3_conv2/Conv2D/ReadVariableOp.^resblock_part2_4_conv1/BiasAdd/ReadVariableOp-^resblock_part2_4_conv1/Conv2D/ReadVariableOp.^resblock_part2_4_conv2/BiasAdd/ReadVariableOp-^resblock_part2_4_conv2/Conv2D/ReadVariableOp.^resblock_part2_5_conv1/BiasAdd/ReadVariableOp-^resblock_part2_5_conv1/Conv2D/ReadVariableOp.^resblock_part2_5_conv2/BiasAdd/ReadVariableOp-^resblock_part2_5_conv2/Conv2D/ReadVariableOp.^resblock_part2_6_conv1/BiasAdd/ReadVariableOp-^resblock_part2_6_conv1/Conv2D/ReadVariableOp.^resblock_part2_6_conv2/BiasAdd/ReadVariableOp-^resblock_part2_6_conv2/Conv2D/ReadVariableOp.^resblock_part2_7_conv1/BiasAdd/ReadVariableOp-^resblock_part2_7_conv1/Conv2D/ReadVariableOp.^resblock_part2_7_conv2/BiasAdd/ReadVariableOp-^resblock_part2_7_conv2/Conv2D/ReadVariableOp.^resblock_part2_8_conv1/BiasAdd/ReadVariableOp-^resblock_part2_8_conv1/Conv2D/ReadVariableOp.^resblock_part2_8_conv2/BiasAdd/ReadVariableOp-^resblock_part2_8_conv2/Conv2D/ReadVariableOp.^resblock_part3_1_conv1/BiasAdd/ReadVariableOp-^resblock_part3_1_conv1/Conv2D/ReadVariableOp.^resblock_part3_1_conv2/BiasAdd/ReadVariableOp-^resblock_part3_1_conv2/Conv2D/ReadVariableOp.^resblock_part3_2_conv1/BiasAdd/ReadVariableOp-^resblock_part3_2_conv1/Conv2D/ReadVariableOp.^resblock_part3_2_conv2/BiasAdd/ReadVariableOp-^resblock_part3_2_conv2/Conv2D/ReadVariableOp.^resblock_part3_3_conv1/BiasAdd/ReadVariableOp-^resblock_part3_3_conv1/Conv2D/ReadVariableOp.^resblock_part3_3_conv2/BiasAdd/ReadVariableOp-^resblock_part3_3_conv2/Conv2D/ReadVariableOp.^resblock_part3_4_conv1/BiasAdd/ReadVariableOp-^resblock_part3_4_conv1/Conv2D/ReadVariableOp.^resblock_part3_4_conv2/BiasAdd/ReadVariableOp-^resblock_part3_4_conv2/Conv2D/ReadVariableOp#^upsampler_1/BiasAdd/ReadVariableOp"^upsampler_1/Conv2D/ReadVariableOp#^upsampler_2/BiasAdd/ReadVariableOp"^upsampler_2/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2L
$downsampler_1/BiasAdd/ReadVariableOp$downsampler_1/BiasAdd/ReadVariableOp2J
#downsampler_1/Conv2D/ReadVariableOp#downsampler_1/Conv2D/ReadVariableOp2L
$downsampler_2/BiasAdd/ReadVariableOp$downsampler_2/BiasAdd/ReadVariableOp2J
#downsampler_2/Conv2D/ReadVariableOp#downsampler_2/Conv2D/ReadVariableOp2F
!extra_conv/BiasAdd/ReadVariableOp!extra_conv/BiasAdd/ReadVariableOp2D
 extra_conv/Conv2D/ReadVariableOp extra_conv/Conv2D/ReadVariableOp2F
!input_conv/BiasAdd/ReadVariableOp!input_conv/BiasAdd/ReadVariableOp2D
 input_conv/Conv2D/ReadVariableOp input_conv/Conv2D/ReadVariableOp2H
"output_conv/BiasAdd/ReadVariableOp"output_conv/BiasAdd/ReadVariableOp2F
!output_conv/Conv2D/ReadVariableOp!output_conv/Conv2D/ReadVariableOp2^
-resblock_part1_1_conv1/BiasAdd/ReadVariableOp-resblock_part1_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_1_conv1/Conv2D/ReadVariableOp,resblock_part1_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_1_conv2/BiasAdd/ReadVariableOp-resblock_part1_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_1_conv2/Conv2D/ReadVariableOp,resblock_part1_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_2_conv1/BiasAdd/ReadVariableOp-resblock_part1_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_2_conv1/Conv2D/ReadVariableOp,resblock_part1_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_2_conv2/BiasAdd/ReadVariableOp-resblock_part1_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_2_conv2/Conv2D/ReadVariableOp,resblock_part1_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_3_conv1/BiasAdd/ReadVariableOp-resblock_part1_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_3_conv1/Conv2D/ReadVariableOp,resblock_part1_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_3_conv2/BiasAdd/ReadVariableOp-resblock_part1_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_3_conv2/Conv2D/ReadVariableOp,resblock_part1_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part1_4_conv1/BiasAdd/ReadVariableOp-resblock_part1_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part1_4_conv1/Conv2D/ReadVariableOp,resblock_part1_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part1_4_conv2/BiasAdd/ReadVariableOp-resblock_part1_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part1_4_conv2/Conv2D/ReadVariableOp,resblock_part1_4_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_1_conv1/BiasAdd/ReadVariableOp-resblock_part2_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_1_conv1/Conv2D/ReadVariableOp,resblock_part2_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_1_conv2/BiasAdd/ReadVariableOp-resblock_part2_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_1_conv2/Conv2D/ReadVariableOp,resblock_part2_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_2_conv1/BiasAdd/ReadVariableOp-resblock_part2_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_2_conv1/Conv2D/ReadVariableOp,resblock_part2_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_2_conv2/BiasAdd/ReadVariableOp-resblock_part2_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_2_conv2/Conv2D/ReadVariableOp,resblock_part2_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_3_conv1/BiasAdd/ReadVariableOp-resblock_part2_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_3_conv1/Conv2D/ReadVariableOp,resblock_part2_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_3_conv2/BiasAdd/ReadVariableOp-resblock_part2_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_3_conv2/Conv2D/ReadVariableOp,resblock_part2_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_4_conv1/BiasAdd/ReadVariableOp-resblock_part2_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_4_conv1/Conv2D/ReadVariableOp,resblock_part2_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_4_conv2/BiasAdd/ReadVariableOp-resblock_part2_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_4_conv2/Conv2D/ReadVariableOp,resblock_part2_4_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_5_conv1/BiasAdd/ReadVariableOp-resblock_part2_5_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_5_conv1/Conv2D/ReadVariableOp,resblock_part2_5_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_5_conv2/BiasAdd/ReadVariableOp-resblock_part2_5_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_5_conv2/Conv2D/ReadVariableOp,resblock_part2_5_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_6_conv1/BiasAdd/ReadVariableOp-resblock_part2_6_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_6_conv1/Conv2D/ReadVariableOp,resblock_part2_6_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_6_conv2/BiasAdd/ReadVariableOp-resblock_part2_6_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_6_conv2/Conv2D/ReadVariableOp,resblock_part2_6_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_7_conv1/BiasAdd/ReadVariableOp-resblock_part2_7_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_7_conv1/Conv2D/ReadVariableOp,resblock_part2_7_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_7_conv2/BiasAdd/ReadVariableOp-resblock_part2_7_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_7_conv2/Conv2D/ReadVariableOp,resblock_part2_7_conv2/Conv2D/ReadVariableOp2^
-resblock_part2_8_conv1/BiasAdd/ReadVariableOp-resblock_part2_8_conv1/BiasAdd/ReadVariableOp2\
,resblock_part2_8_conv1/Conv2D/ReadVariableOp,resblock_part2_8_conv1/Conv2D/ReadVariableOp2^
-resblock_part2_8_conv2/BiasAdd/ReadVariableOp-resblock_part2_8_conv2/BiasAdd/ReadVariableOp2\
,resblock_part2_8_conv2/Conv2D/ReadVariableOp,resblock_part2_8_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_1_conv1/BiasAdd/ReadVariableOp-resblock_part3_1_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_1_conv1/Conv2D/ReadVariableOp,resblock_part3_1_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_1_conv2/BiasAdd/ReadVariableOp-resblock_part3_1_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_1_conv2/Conv2D/ReadVariableOp,resblock_part3_1_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_2_conv1/BiasAdd/ReadVariableOp-resblock_part3_2_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_2_conv1/Conv2D/ReadVariableOp,resblock_part3_2_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_2_conv2/BiasAdd/ReadVariableOp-resblock_part3_2_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_2_conv2/Conv2D/ReadVariableOp,resblock_part3_2_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_3_conv1/BiasAdd/ReadVariableOp-resblock_part3_3_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_3_conv1/Conv2D/ReadVariableOp,resblock_part3_3_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_3_conv2/BiasAdd/ReadVariableOp-resblock_part3_3_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_3_conv2/Conv2D/ReadVariableOp,resblock_part3_3_conv2/Conv2D/ReadVariableOp2^
-resblock_part3_4_conv1/BiasAdd/ReadVariableOp-resblock_part3_4_conv1/BiasAdd/ReadVariableOp2\
,resblock_part3_4_conv1/Conv2D/ReadVariableOp,resblock_part3_4_conv1/Conv2D/ReadVariableOp2^
-resblock_part3_4_conv2/BiasAdd/ReadVariableOp-resblock_part3_4_conv2/BiasAdd/ReadVariableOp2\
,resblock_part3_4_conv2/Conv2D/ReadVariableOp,resblock_part3_4_conv2/Conv2D/ReadVariableOp2H
"upsampler_1/BiasAdd/ReadVariableOp"upsampler_1/BiasAdd/ReadVariableOp2F
!upsampler_1/Conv2D/ReadVariableOp!upsampler_1/Conv2D/ReadVariableOp2H
"upsampler_2/BiasAdd/ReadVariableOp"upsampler_2/BiasAdd/ReadVariableOp2F
!upsampler_2/Conv2D/ReadVariableOp!upsampler_2/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
¢
ÑT
__inference__wrapped_model_2058
input_layer:
6ssi_res_unet_input_conv_conv2d_readvariableop_resource;
7ssi_res_unet_input_conv_biasadd_readvariableop_resource=
9ssi_res_unet_downsampler_1_conv2d_readvariableop_resource>
:ssi_res_unet_downsampler_1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part1_1_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_1_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part1_1_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_1_conv2_biasadd_readvariableop_resource'
#ssi_res_unet_tf_math_multiply_mul_xF
Bssi_res_unet_resblock_part1_2_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_2_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part1_2_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_2_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_1_mul_xF
Bssi_res_unet_resblock_part1_3_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_3_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part1_3_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_3_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_2_mul_xF
Bssi_res_unet_resblock_part1_4_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_4_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part1_4_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part1_4_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_3_mul_x=
9ssi_res_unet_downsampler_2_conv2d_readvariableop_resource>
:ssi_res_unet_downsampler_2_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_1_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_1_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_1_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_1_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_4_mul_xF
Bssi_res_unet_resblock_part2_2_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_2_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_2_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_2_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_5_mul_xF
Bssi_res_unet_resblock_part2_3_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_3_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_3_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_3_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_6_mul_xF
Bssi_res_unet_resblock_part2_4_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_4_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_4_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_4_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_7_mul_xF
Bssi_res_unet_resblock_part2_5_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_5_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_5_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_5_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_8_mul_xF
Bssi_res_unet_resblock_part2_6_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_6_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_6_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_6_conv2_biasadd_readvariableop_resource)
%ssi_res_unet_tf_math_multiply_9_mul_xF
Bssi_res_unet_resblock_part2_7_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_7_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_7_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_7_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_10_mul_xF
Bssi_res_unet_resblock_part2_8_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_8_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part2_8_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part2_8_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_11_mul_x;
7ssi_res_unet_upsampler_1_conv2d_readvariableop_resource<
8ssi_res_unet_upsampler_1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part3_1_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_1_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part3_1_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_1_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_12_mul_xF
Bssi_res_unet_resblock_part3_2_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_2_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part3_2_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_2_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_13_mul_xF
Bssi_res_unet_resblock_part3_3_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_3_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part3_3_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_3_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_14_mul_xF
Bssi_res_unet_resblock_part3_4_conv1_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_4_conv1_biasadd_readvariableop_resourceF
Bssi_res_unet_resblock_part3_4_conv2_conv2d_readvariableop_resourceG
Cssi_res_unet_resblock_part3_4_conv2_biasadd_readvariableop_resource*
&ssi_res_unet_tf_math_multiply_15_mul_x:
6ssi_res_unet_extra_conv_conv2d_readvariableop_resource;
7ssi_res_unet_extra_conv_biasadd_readvariableop_resource;
7ssi_res_unet_upsampler_2_conv2d_readvariableop_resource<
8ssi_res_unet_upsampler_2_biasadd_readvariableop_resource;
7ssi_res_unet_output_conv_conv2d_readvariableop_resource<
8ssi_res_unet_output_conv_biasadd_readvariableop_resource
identity¢1ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp¢0ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp¢1ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp¢0ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp¢.ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp¢-ssi_res_unet/extra_conv/Conv2D/ReadVariableOp¢.ssi_res_unet/input_conv/BiasAdd/ReadVariableOp¢-ssi_res_unet/input_conv/Conv2D/ReadVariableOp¢/ssi_res_unet/output_conv/BiasAdd/ReadVariableOp¢.ssi_res_unet/output_conv/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOp¢:ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp¢9ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOp¢/ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp¢.ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp¢/ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp¢.ssi_res_unet/upsampler_2/Conv2D/ReadVariableOpÝ
-ssi_res_unet/input_conv/Conv2D/ReadVariableOpReadVariableOp6ssi_res_unet_input_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02/
-ssi_res_unet/input_conv/Conv2D/ReadVariableOp
ssi_res_unet/input_conv/Conv2DConv2Dinput_layer5ssi_res_unet/input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2 
ssi_res_unet/input_conv/Conv2DÔ
.ssi_res_unet/input_conv/BiasAdd/ReadVariableOpReadVariableOp7ssi_res_unet_input_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.ssi_res_unet/input_conv/BiasAdd/ReadVariableOp
ssi_res_unet/input_conv/BiasAddBiasAdd'ssi_res_unet/input_conv/Conv2D:output:06ssi_res_unet/input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2!
ssi_res_unet/input_conv/BiasAddÅ
(ssi_res_unet/zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2*
(ssi_res_unet/zero_padding2d/Pad/paddingsâ
ssi_res_unet/zero_padding2d/PadPad(ssi_res_unet/input_conv/BiasAdd:output:01ssi_res_unet/zero_padding2d/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
ssi_res_unet/zero_padding2d/Padæ
0ssi_res_unet/downsampler_1/Conv2D/ReadVariableOpReadVariableOp9ssi_res_unet_downsampler_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp°
!ssi_res_unet/downsampler_1/Conv2DConv2D(ssi_res_unet/zero_padding2d/Pad:output:08ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingVALID*
strides
2#
!ssi_res_unet/downsampler_1/Conv2DÝ
1ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOpReadVariableOp:ssi_res_unet_downsampler_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp
"ssi_res_unet/downsampler_1/BiasAddBiasAdd*ssi_res_unet/downsampler_1/Conv2D:output:09ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2$
"ssi_res_unet/downsampler_1/BiasAdd
9ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part1_1_conv1/Conv2DConv2D+ssi_res_unet/downsampler_1/BiasAdd:output:0Assi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_1_conv1/Conv2Dø
:ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_1_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part1_1_conv1/Conv2D:output:0Bssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_1_conv1/BiasAddÎ
(ssi_res_unet/resblock_part1_1_relu1/ReluRelu4ssi_res_unet/resblock_part1_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part1_1_relu1/Relu
9ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part1_1_conv2/Conv2DConv2D6ssi_res_unet/resblock_part1_1_relu1/Relu:activations:0Assi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_1_conv2/Conv2Dø
:ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_1_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part1_1_conv2/Conv2D:output:0Bssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_1_conv2/BiasAddä
!ssi_res_unet/tf.math.multiply/MulMul#ssi_res_unet_tf_math_multiply_mul_x4ssi_res_unet/resblock_part1_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!ssi_res_unet/tf.math.multiply/Mulë
'ssi_res_unet/tf.__operators__.add/AddV2AddV2%ssi_res_unet/tf.math.multiply/Mul:z:0+ssi_res_unet/downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'ssi_res_unet/tf.__operators__.add/AddV2
9ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part1_2_conv1/Conv2DConv2D+ssi_res_unet/tf.__operators__.add/AddV2:z:0Assi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_2_conv1/Conv2Dø
:ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_2_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part1_2_conv1/Conv2D:output:0Bssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_2_conv1/BiasAddÎ
(ssi_res_unet/resblock_part1_2_relu1/ReluRelu4ssi_res_unet/resblock_part1_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part1_2_relu1/Relu
9ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part1_2_conv2/Conv2DConv2D6ssi_res_unet/resblock_part1_2_relu1/Relu:activations:0Assi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_2_conv2/Conv2Dø
:ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_2_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part1_2_conv2/Conv2D:output:0Bssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_2_conv2/BiasAddê
#ssi_res_unet/tf.math.multiply_1/MulMul%ssi_res_unet_tf_math_multiply_1_mul_x4ssi_res_unet/resblock_part1_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#ssi_res_unet/tf.math.multiply_1/Mulñ
)ssi_res_unet/tf.__operators__.add_1/AddV2AddV2'ssi_res_unet/tf.math.multiply_1/Mul:z:0+ssi_res_unet/tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)ssi_res_unet/tf.__operators__.add_1/AddV2
9ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOpÏ
*ssi_res_unet/resblock_part1_3_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_1/AddV2:z:0Assi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_3_conv1/Conv2Dø
:ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_3_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part1_3_conv1/Conv2D:output:0Bssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_3_conv1/BiasAddÎ
(ssi_res_unet/resblock_part1_3_relu1/ReluRelu4ssi_res_unet/resblock_part1_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part1_3_relu1/Relu
9ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part1_3_conv2/Conv2DConv2D6ssi_res_unet/resblock_part1_3_relu1/Relu:activations:0Assi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_3_conv2/Conv2Dø
:ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_3_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part1_3_conv2/Conv2D:output:0Bssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_3_conv2/BiasAddê
#ssi_res_unet/tf.math.multiply_2/MulMul%ssi_res_unet_tf_math_multiply_2_mul_x4ssi_res_unet/resblock_part1_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#ssi_res_unet/tf.math.multiply_2/Muló
)ssi_res_unet/tf.__operators__.add_2/AddV2AddV2'ssi_res_unet/tf.math.multiply_2/Mul:z:0-ssi_res_unet/tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)ssi_res_unet/tf.__operators__.add_2/AddV2
9ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOpÏ
*ssi_res_unet/resblock_part1_4_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_2/AddV2:z:0Assi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_4_conv1/Conv2Dø
:ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_4_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part1_4_conv1/Conv2D:output:0Bssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_4_conv1/BiasAddÎ
(ssi_res_unet/resblock_part1_4_relu1/ReluRelu4ssi_res_unet/resblock_part1_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part1_4_relu1/Relu
9ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part1_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part1_4_conv2/Conv2DConv2D6ssi_res_unet/resblock_part1_4_relu1/Relu:activations:0Assi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part1_4_conv2/Conv2Dø
:ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part1_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part1_4_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part1_4_conv2/Conv2D:output:0Bssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part1_4_conv2/BiasAddê
#ssi_res_unet/tf.math.multiply_3/MulMul%ssi_res_unet_tf_math_multiply_3_mul_x4ssi_res_unet/resblock_part1_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#ssi_res_unet/tf.math.multiply_3/Muló
)ssi_res_unet/tf.__operators__.add_3/AddV2AddV2'ssi_res_unet/tf.math.multiply_3/Mul:z:0-ssi_res_unet/tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)ssi_res_unet/tf.__operators__.add_3/AddV2É
*ssi_res_unet/zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2,
*ssi_res_unet/zero_padding2d_1/Pad/paddingsí
!ssi_res_unet/zero_padding2d_1/PadPad-ssi_res_unet/tf.__operators__.add_3/AddV2:z:03ssi_res_unet/zero_padding2d_1/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!ssi_res_unet/zero_padding2d_1/Padæ
0ssi_res_unet/downsampler_2/Conv2D/ReadVariableOpReadVariableOp9ssi_res_unet_downsampler_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp°
!ssi_res_unet/downsampler_2/Conv2DConv2D*ssi_res_unet/zero_padding2d_1/Pad:output:08ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingVALID*
strides
2#
!ssi_res_unet/downsampler_2/Conv2DÝ
1ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOpReadVariableOp:ssi_res_unet_downsampler_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp
"ssi_res_unet/downsampler_2/BiasAddBiasAdd*ssi_res_unet/downsampler_2/Conv2D:output:09ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2$
"ssi_res_unet/downsampler_2/BiasAdd
9ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOpË
*ssi_res_unet/resblock_part2_1_conv1/Conv2DConv2D+ssi_res_unet/downsampler_2/BiasAdd:output:0Assi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_1_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_1_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_1_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_1_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_1_relu1/ReluRelu4ssi_res_unet/resblock_part2_1_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_1_relu1/Relu
9ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_1_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_1_relu1/Relu:activations:0Assi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_1_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_1_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_1_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_1_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_4/MulMul%ssi_res_unet_tf_math_multiply_4_mul_x4ssi_res_unet/resblock_part2_1_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_4/Mulï
)ssi_res_unet/tf.__operators__.add_4/AddV2AddV2'ssi_res_unet/tf.math.multiply_4/Mul:z:0+ssi_res_unet/downsampler_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_4/AddV2
9ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_2_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_4/AddV2:z:0Assi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_2_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_2_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_2_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_2_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_2_relu1/ReluRelu4ssi_res_unet/resblock_part2_2_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_2_relu1/Relu
9ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_2_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_2_relu1/Relu:activations:0Assi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_2_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_2_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_2_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_2_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_5/MulMul%ssi_res_unet_tf_math_multiply_5_mul_x4ssi_res_unet/resblock_part2_2_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_5/Mulñ
)ssi_res_unet/tf.__operators__.add_5/AddV2AddV2'ssi_res_unet/tf.math.multiply_5/Mul:z:0-ssi_res_unet/tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_5/AddV2
9ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_3_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_5/AddV2:z:0Assi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_3_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_3_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_3_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_3_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_3_relu1/ReluRelu4ssi_res_unet/resblock_part2_3_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_3_relu1/Relu
9ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_3_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_3_relu1/Relu:activations:0Assi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_3_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_3_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_3_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_3_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_6/MulMul%ssi_res_unet_tf_math_multiply_6_mul_x4ssi_res_unet/resblock_part2_3_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_6/Mulñ
)ssi_res_unet/tf.__operators__.add_6/AddV2AddV2'ssi_res_unet/tf.math.multiply_6/Mul:z:0-ssi_res_unet/tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_6/AddV2
9ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_4_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_6/AddV2:z:0Assi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_4_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_4_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_4_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_4_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_4_relu1/ReluRelu4ssi_res_unet/resblock_part2_4_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_4_relu1/Relu
9ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_4_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_4_relu1/Relu:activations:0Assi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_4_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_4_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_4_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_4_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_7/MulMul%ssi_res_unet_tf_math_multiply_7_mul_x4ssi_res_unet/resblock_part2_4_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_7/Mulñ
)ssi_res_unet/tf.__operators__.add_7/AddV2AddV2'ssi_res_unet/tf.math.multiply_7/Mul:z:0-ssi_res_unet/tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_7/AddV2
9ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_5_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_5_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_7/AddV2:z:0Assi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_5_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_5_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_5_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_5_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_5_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_5_relu1/ReluRelu4ssi_res_unet/resblock_part2_5_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_5_relu1/Relu
9ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_5_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_5_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_5_relu1/Relu:activations:0Assi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_5_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_5_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_5_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_5_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_5_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_8/MulMul%ssi_res_unet_tf_math_multiply_8_mul_x4ssi_res_unet/resblock_part2_5_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_8/Mulñ
)ssi_res_unet/tf.__operators__.add_8/AddV2AddV2'ssi_res_unet/tf.math.multiply_8/Mul:z:0-ssi_res_unet/tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_8/AddV2
9ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_6_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_6_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_8/AddV2:z:0Assi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_6_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_6_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_6_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_6_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_6_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_6_relu1/ReluRelu4ssi_res_unet/resblock_part2_6_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_6_relu1/Relu
9ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_6_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_6_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_6_relu1/Relu:activations:0Assi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_6_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_6_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_6_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_6_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_6_conv2/BiasAddè
#ssi_res_unet/tf.math.multiply_9/MulMul%ssi_res_unet_tf_math_multiply_9_mul_x4ssi_res_unet/resblock_part2_6_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2%
#ssi_res_unet/tf.math.multiply_9/Mulñ
)ssi_res_unet/tf.__operators__.add_9/AddV2AddV2'ssi_res_unet/tf.math.multiply_9/Mul:z:0-ssi_res_unet/tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2+
)ssi_res_unet/tf.__operators__.add_9/AddV2
9ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_7_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOpÍ
*ssi_res_unet/resblock_part2_7_conv1/Conv2DConv2D-ssi_res_unet/tf.__operators__.add_9/AddV2:z:0Assi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_7_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_7_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_7_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_7_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_7_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_7_relu1/ReluRelu4ssi_res_unet/resblock_part2_7_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_7_relu1/Relu
9ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_7_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_7_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_7_relu1/Relu:activations:0Assi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_7_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_7_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_7_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_7_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_7_conv2/BiasAddë
$ssi_res_unet/tf.math.multiply_10/MulMul&ssi_res_unet_tf_math_multiply_10_mul_x4ssi_res_unet/resblock_part2_7_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2&
$ssi_res_unet/tf.math.multiply_10/Mulô
*ssi_res_unet/tf.__operators__.add_10/AddV2AddV2(ssi_res_unet/tf.math.multiply_10/Mul:z:0-ssi_res_unet/tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2,
*ssi_res_unet/tf.__operators__.add_10/AddV2
9ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_8_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOpÎ
*ssi_res_unet/resblock_part2_8_conv1/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_10/AddV2:z:0Assi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_8_conv1/Conv2Dø
:ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_8_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_8_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part2_8_conv1/Conv2D:output:0Bssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_8_conv1/BiasAddÌ
(ssi_res_unet/resblock_part2_8_relu1/ReluRelu4ssi_res_unet/resblock_part2_8_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2*
(ssi_res_unet/resblock_part2_8_relu1/Relu
9ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part2_8_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOpÖ
*ssi_res_unet/resblock_part2_8_conv2/Conv2DConv2D6ssi_res_unet/resblock_part2_8_relu1/Relu:activations:0Assi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part2_8_conv2/Conv2Dø
:ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part2_8_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp¯
+ssi_res_unet/resblock_part2_8_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part2_8_conv2/Conv2D:output:0Bssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2-
+ssi_res_unet/resblock_part2_8_conv2/BiasAddë
$ssi_res_unet/tf.math.multiply_11/MulMul&ssi_res_unet_tf_math_multiply_11_mul_x4ssi_res_unet/resblock_part2_8_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2&
$ssi_res_unet/tf.math.multiply_11/Mulõ
*ssi_res_unet/tf.__operators__.add_11/AddV2AddV2(ssi_res_unet/tf.math.multiply_11/Mul:z:0.ssi_res_unet/tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2,
*ssi_res_unet/tf.__operators__.add_11/AddV2á
.ssi_res_unet/upsampler_1/Conv2D/ReadVariableOpReadVariableOp7ssi_res_unet_upsampler_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype020
.ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp®
ssi_res_unet/upsampler_1/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_11/AddV2:z:06ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW*
paddingSAME*
strides
2!
ssi_res_unet/upsampler_1/Conv2DØ
/ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOpReadVariableOp8ssi_res_unet_upsampler_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp
 ssi_res_unet/upsampler_1/BiasAddBiasAdd(ssi_res_unet/upsampler_1/Conv2D:output:07ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW2"
 ssi_res_unet/upsampler_1/BiasAdd
.ssi_res_unet/tf.nn.depth_to_space/DepthToSpaceDepthToSpace)ssi_res_unet/upsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW20
.ssi_res_unet/tf.nn.depth_to_space/DepthToSpace
9ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOpÙ
*ssi_res_unet/resblock_part3_1_conv1/Conv2DConv2D7ssi_res_unet/tf.nn.depth_to_space/DepthToSpace:output:0Assi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_1_conv1/Conv2Dø
:ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_1_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part3_1_conv1/Conv2D:output:0Bssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_1_conv1/BiasAddÎ
(ssi_res_unet/resblock_part3_1_relu1/ReluRelu4ssi_res_unet/resblock_part3_1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part3_1_relu1/Relu
9ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part3_1_conv2/Conv2DConv2D6ssi_res_unet/resblock_part3_1_relu1/Relu:activations:0Assi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_1_conv2/Conv2Dø
:ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_1_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part3_1_conv2/Conv2D:output:0Bssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_1_conv2/BiasAddí
$ssi_res_unet/tf.math.multiply_12/MulMul&ssi_res_unet_tf_math_multiply_12_mul_x4ssi_res_unet/resblock_part3_1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$ssi_res_unet/tf.math.multiply_12/Mul
*ssi_res_unet/tf.__operators__.add_12/AddV2AddV2(ssi_res_unet/tf.math.multiply_12/Mul:z:07ssi_res_unet/tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*ssi_res_unet/tf.__operators__.add_12/AddV2
9ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOpÐ
*ssi_res_unet/resblock_part3_2_conv1/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_12/AddV2:z:0Assi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_2_conv1/Conv2Dø
:ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_2_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_2_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part3_2_conv1/Conv2D:output:0Bssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_2_conv1/BiasAddÎ
(ssi_res_unet/resblock_part3_2_relu1/ReluRelu4ssi_res_unet/resblock_part3_2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part3_2_relu1/Relu
9ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_2_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part3_2_conv2/Conv2DConv2D6ssi_res_unet/resblock_part3_2_relu1/Relu:activations:0Assi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_2_conv2/Conv2Dø
:ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_2_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_2_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part3_2_conv2/Conv2D:output:0Bssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_2_conv2/BiasAddí
$ssi_res_unet/tf.math.multiply_13/MulMul&ssi_res_unet_tf_math_multiply_13_mul_x4ssi_res_unet/resblock_part3_2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$ssi_res_unet/tf.math.multiply_13/Mul÷
*ssi_res_unet/tf.__operators__.add_13/AddV2AddV2(ssi_res_unet/tf.math.multiply_13/Mul:z:0.ssi_res_unet/tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*ssi_res_unet/tf.__operators__.add_13/AddV2
9ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOpÐ
*ssi_res_unet/resblock_part3_3_conv1/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_13/AddV2:z:0Assi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_3_conv1/Conv2Dø
:ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_3_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_3_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part3_3_conv1/Conv2D:output:0Bssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_3_conv1/BiasAddÎ
(ssi_res_unet/resblock_part3_3_relu1/ReluRelu4ssi_res_unet/resblock_part3_3_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part3_3_relu1/Relu
9ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_3_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part3_3_conv2/Conv2DConv2D6ssi_res_unet/resblock_part3_3_relu1/Relu:activations:0Assi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_3_conv2/Conv2Dø
:ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_3_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_3_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part3_3_conv2/Conv2D:output:0Bssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_3_conv2/BiasAddí
$ssi_res_unet/tf.math.multiply_14/MulMul&ssi_res_unet_tf_math_multiply_14_mul_x4ssi_res_unet/resblock_part3_3_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$ssi_res_unet/tf.math.multiply_14/Mul÷
*ssi_res_unet/tf.__operators__.add_14/AddV2AddV2(ssi_res_unet/tf.math.multiply_14/Mul:z:0.ssi_res_unet/tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*ssi_res_unet/tf.__operators__.add_14/AddV2
9ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_4_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOpÐ
*ssi_res_unet/resblock_part3_4_conv1/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_14/AddV2:z:0Assi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_4_conv1/Conv2Dø
:ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_4_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_4_conv1/BiasAddBiasAdd3ssi_res_unet/resblock_part3_4_conv1/Conv2D:output:0Bssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_4_conv1/BiasAddÎ
(ssi_res_unet/resblock_part3_4_relu1/ReluRelu4ssi_res_unet/resblock_part3_4_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(ssi_res_unet/resblock_part3_4_relu1/Relu
9ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOpReadVariableOpBssi_res_unet_resblock_part3_4_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOpØ
*ssi_res_unet/resblock_part3_4_conv2/Conv2DConv2D6ssi_res_unet/resblock_part3_4_relu1/Relu:activations:0Assi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2,
*ssi_res_unet/resblock_part3_4_conv2/Conv2Dø
:ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOpReadVariableOpCssi_res_unet_resblock_part3_4_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp±
+ssi_res_unet/resblock_part3_4_conv2/BiasAddBiasAdd3ssi_res_unet/resblock_part3_4_conv2/Conv2D:output:0Bssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2-
+ssi_res_unet/resblock_part3_4_conv2/BiasAddí
$ssi_res_unet/tf.math.multiply_15/MulMul&ssi_res_unet_tf_math_multiply_15_mul_x4ssi_res_unet/resblock_part3_4_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$ssi_res_unet/tf.math.multiply_15/Mul÷
*ssi_res_unet/tf.__operators__.add_15/AddV2AddV2(ssi_res_unet/tf.math.multiply_15/Mul:z:0.ssi_res_unet/tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*ssi_res_unet/tf.__operators__.add_15/AddV2Ý
-ssi_res_unet/extra_conv/Conv2D/ReadVariableOpReadVariableOp6ssi_res_unet_extra_conv_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-ssi_res_unet/extra_conv/Conv2D/ReadVariableOp¬
ssi_res_unet/extra_conv/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_15/AddV2:z:05ssi_res_unet/extra_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2 
ssi_res_unet/extra_conv/Conv2DÔ
.ssi_res_unet/extra_conv/BiasAdd/ReadVariableOpReadVariableOp7ssi_res_unet_extra_conv_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp
ssi_res_unet/extra_conv/BiasAddBiasAdd'ssi_res_unet/extra_conv/Conv2D:output:06ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2!
ssi_res_unet/extra_conv/BiasAddô
*ssi_res_unet/tf.__operators__.add_16/AddV2AddV2(ssi_res_unet/extra_conv/BiasAdd:output:0+ssi_res_unet/downsampler_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*ssi_res_unet/tf.__operators__.add_16/AddV2á
.ssi_res_unet/upsampler_2/Conv2D/ReadVariableOpReadVariableOp7ssi_res_unet_upsampler_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype020
.ssi_res_unet/upsampler_2/Conv2D/ReadVariableOp°
ssi_res_unet/upsampler_2/Conv2DConv2D.ssi_res_unet/tf.__operators__.add_16/AddV2:z:06ssi_res_unet/upsampler_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2!
ssi_res_unet/upsampler_2/Conv2DØ
/ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOpReadVariableOp8ssi_res_unet_upsampler_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp
 ssi_res_unet/upsampler_2/BiasAddBiasAdd(ssi_res_unet/upsampler_2/Conv2D:output:07ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2"
 ssi_res_unet/upsampler_2/BiasAdd
0ssi_res_unet/tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace)ssi_res_unet/upsampler_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW22
0ssi_res_unet/tf.nn.depth_to_space_1/DepthToSpaceà
.ssi_res_unet/output_conv/Conv2D/ReadVariableOpReadVariableOp7ssi_res_unet_output_conv_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype020
.ssi_res_unet/output_conv/Conv2D/ReadVariableOpº
ssi_res_unet/output_conv/Conv2DConv2D9ssi_res_unet/tf.nn.depth_to_space_1/DepthToSpace:output:06ssi_res_unet/output_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2!
ssi_res_unet/output_conv/Conv2D×
/ssi_res_unet/output_conv/BiasAdd/ReadVariableOpReadVariableOp8ssi_res_unet_output_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/ssi_res_unet/output_conv/BiasAdd/ReadVariableOp
 ssi_res_unet/output_conv/BiasAddBiasAdd(ssi_res_unet/output_conv/Conv2D:output:07ssi_res_unet/output_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2"
 ssi_res_unet/output_conv/BiasAddà$
IdentityIdentity)ssi_res_unet/output_conv/BiasAdd:output:02^ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp1^ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp2^ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp1^ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp/^ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp.^ssi_res_unet/extra_conv/Conv2D/ReadVariableOp/^ssi_res_unet/input_conv/BiasAdd/ReadVariableOp.^ssi_res_unet/input_conv/Conv2D/ReadVariableOp0^ssi_res_unet/output_conv/BiasAdd/ReadVariableOp/^ssi_res_unet/output_conv/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOp;^ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp:^ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOp0^ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp/^ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp0^ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp/^ssi_res_unet/upsampler_2/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2f
1ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp1ssi_res_unet/downsampler_1/BiasAdd/ReadVariableOp2d
0ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp0ssi_res_unet/downsampler_1/Conv2D/ReadVariableOp2f
1ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp1ssi_res_unet/downsampler_2/BiasAdd/ReadVariableOp2d
0ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp0ssi_res_unet/downsampler_2/Conv2D/ReadVariableOp2`
.ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp.ssi_res_unet/extra_conv/BiasAdd/ReadVariableOp2^
-ssi_res_unet/extra_conv/Conv2D/ReadVariableOp-ssi_res_unet/extra_conv/Conv2D/ReadVariableOp2`
.ssi_res_unet/input_conv/BiasAdd/ReadVariableOp.ssi_res_unet/input_conv/BiasAdd/ReadVariableOp2^
-ssi_res_unet/input_conv/Conv2D/ReadVariableOp-ssi_res_unet/input_conv/Conv2D/ReadVariableOp2b
/ssi_res_unet/output_conv/BiasAdd/ReadVariableOp/ssi_res_unet/output_conv/BiasAdd/ReadVariableOp2`
.ssi_res_unet/output_conv/Conv2D/ReadVariableOp.ssi_res_unet/output_conv/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_1_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_1_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_1_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_1_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_2_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_2_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_2_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_2_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_3_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_3_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_3_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_3_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_4_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_4_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part1_4_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part1_4_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_1_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_1_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_1_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_1_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_2_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_2_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_2_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_2_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_3_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_3_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_3_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_3_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_4_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_4_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_4_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_4_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_5_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_5_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_5_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_5_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_6_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_6_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_6_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_6_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_7_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_7_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_7_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_7_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_8_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_8_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part2_8_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part2_8_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_1_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_1_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_1_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_1_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_2_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_2_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_2_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_2_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_3_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_3_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_3_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_3_conv2/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_4_conv1/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_4_conv1/Conv2D/ReadVariableOp2x
:ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp:ssi_res_unet/resblock_part3_4_conv2/BiasAdd/ReadVariableOp2v
9ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOp9ssi_res_unet/resblock_part3_4_conv2/Conv2D/ReadVariableOp2b
/ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp/ssi_res_unet/upsampler_1/BiasAdd/ReadVariableOp2`
.ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp.ssi_res_unet/upsampler_1/Conv2D/ReadVariableOp2b
/ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp/ssi_res_unet/upsampler_2/BiasAdd/ReadVariableOp2`
.ssi_res_unet/upsampler_2/Conv2D/ReadVariableOp.ssi_res_unet/upsampler_2/Conv2D/ReadVariableOp:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
é
í%
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_3903

inputs
input_conv_3638
input_conv_3640
downsampler_1_3644
downsampler_1_3646
resblock_part1_1_conv1_3649
resblock_part1_1_conv1_3651
resblock_part1_1_conv2_3655
resblock_part1_1_conv2_3657
tf_math_multiply_mul_x
resblock_part1_2_conv1_3663
resblock_part1_2_conv1_3665
resblock_part1_2_conv2_3669
resblock_part1_2_conv2_3671
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_3677
resblock_part1_3_conv1_3679
resblock_part1_3_conv2_3683
resblock_part1_3_conv2_3685
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_3691
resblock_part1_4_conv1_3693
resblock_part1_4_conv2_3697
resblock_part1_4_conv2_3699
tf_math_multiply_3_mul_x
downsampler_2_3706
downsampler_2_3708
resblock_part2_1_conv1_3711
resblock_part2_1_conv1_3713
resblock_part2_1_conv2_3717
resblock_part2_1_conv2_3719
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_3725
resblock_part2_2_conv1_3727
resblock_part2_2_conv2_3731
resblock_part2_2_conv2_3733
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_3739
resblock_part2_3_conv1_3741
resblock_part2_3_conv2_3745
resblock_part2_3_conv2_3747
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_3753
resblock_part2_4_conv1_3755
resblock_part2_4_conv2_3759
resblock_part2_4_conv2_3761
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_3767
resblock_part2_5_conv1_3769
resblock_part2_5_conv2_3773
resblock_part2_5_conv2_3775
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_3781
resblock_part2_6_conv1_3783
resblock_part2_6_conv2_3787
resblock_part2_6_conv2_3789
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_3795
resblock_part2_7_conv1_3797
resblock_part2_7_conv2_3801
resblock_part2_7_conv2_3803
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_3809
resblock_part2_8_conv1_3811
resblock_part2_8_conv2_3815
resblock_part2_8_conv2_3817
tf_math_multiply_11_mul_x
upsampler_1_3823
upsampler_1_3825
resblock_part3_1_conv1_3829
resblock_part3_1_conv1_3831
resblock_part3_1_conv2_3835
resblock_part3_1_conv2_3837
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_3843
resblock_part3_2_conv1_3845
resblock_part3_2_conv2_3849
resblock_part3_2_conv2_3851
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_3857
resblock_part3_3_conv1_3859
resblock_part3_3_conv2_3863
resblock_part3_3_conv2_3865
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_3871
resblock_part3_4_conv1_3873
resblock_part3_4_conv2_3877
resblock_part3_4_conv2_3879
tf_math_multiply_15_mul_x
extra_conv_3885
extra_conv_3887
upsampler_2_3891
upsampler_2_3893
output_conv_3897
output_conv_3899
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCall¥
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_3638input_conv_3640*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_input_conv_layer_call_and_return_conditional_losses_20982$
"input_conv/StatefulPartitionedCall
zero_padding2d/PartitionedCallPartitionedCall+input_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_20652 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_3644downsampler_1_3646*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_1_layer_call_and_return_conditional_losses_21252'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_3649resblock_part1_1_conv1_3651*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_215120
.resblock_part1_1_conv1/StatefulPartitionedCallº
&resblock_part1_1_relu1/PartitionedCallPartitionedCall7resblock_part1_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_21722(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_3655resblock_part1_1_conv2_3657*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_219020
.resblock_part1_1_conv2/StatefulPartitionedCallÀ
tf.math.multiply/MulMultf_math_multiply_mul_x7resblock_part1_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/MulÇ
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2ù
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_3663resblock_part1_2_conv1_3665*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_221920
.resblock_part1_2_conv1/StatefulPartitionedCallº
&resblock_part1_2_relu1/PartitionedCallPartitionedCall7resblock_part1_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_22402(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_3669resblock_part1_2_conv2_3671*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_225820
.resblock_part1_2_conv2/StatefulPartitionedCallÆ
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x7resblock_part1_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2û
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_3677resblock_part1_3_conv1_3679*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_228720
.resblock_part1_3_conv1/StatefulPartitionedCallº
&resblock_part1_3_relu1/PartitionedCallPartitionedCall7resblock_part1_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_23082(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_3683resblock_part1_3_conv2_3685*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_232620
.resblock_part1_3_conv2/StatefulPartitionedCallÆ
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x7resblock_part1_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2û
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_3691resblock_part1_4_conv1_3693*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_235520
.resblock_part1_4_conv1/StatefulPartitionedCallº
&resblock_part1_4_relu1/PartitionedCallPartitionedCall7resblock_part1_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_23762(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_3697resblock_part1_4_conv2_3699*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_239420
.resblock_part1_4_conv2/StatefulPartitionedCallÆ
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x7resblock_part1_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2
 zero_padding2d_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_20782"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_3706downsampler_2_3708*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_2_layer_call_and_return_conditional_losses_24242'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_3711resblock_part2_1_conv1_3713*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_245020
.resblock_part2_1_conv1/StatefulPartitionedCall¸
&resblock_part2_1_relu1/PartitionedCallPartitionedCall7resblock_part2_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_24712(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_3717resblock_part2_1_conv2_3719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_248920
.resblock_part2_1_conv2/StatefulPartitionedCallÄ
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x7resblock_part2_1_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/MulË
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0.downsampler_2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2ù
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_3725resblock_part2_2_conv1_3727*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_251820
.resblock_part2_2_conv1/StatefulPartitionedCall¸
&resblock_part2_2_relu1/PartitionedCallPartitionedCall7resblock_part2_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_25392(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_3731resblock_part2_2_conv2_3733*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_255720
.resblock_part2_2_conv2/StatefulPartitionedCallÄ
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x7resblock_part2_2_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2ù
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_3739resblock_part2_3_conv1_3741*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_258620
.resblock_part2_3_conv1/StatefulPartitionedCall¸
&resblock_part2_3_relu1/PartitionedCallPartitionedCall7resblock_part2_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_26072(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_3745resblock_part2_3_conv2_3747*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_262520
.resblock_part2_3_conv2/StatefulPartitionedCallÄ
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x7resblock_part2_3_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2ù
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_3753resblock_part2_4_conv1_3755*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_265420
.resblock_part2_4_conv1/StatefulPartitionedCall¸
&resblock_part2_4_relu1/PartitionedCallPartitionedCall7resblock_part2_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_26752(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_3759resblock_part2_4_conv2_3761*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_269320
.resblock_part2_4_conv2/StatefulPartitionedCallÄ
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x7resblock_part2_4_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2ù
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_3767resblock_part2_5_conv1_3769*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_272220
.resblock_part2_5_conv1/StatefulPartitionedCall¸
&resblock_part2_5_relu1/PartitionedCallPartitionedCall7resblock_part2_5_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_27432(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_3773resblock_part2_5_conv2_3775*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_276120
.resblock_part2_5_conv2/StatefulPartitionedCallÄ
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x7resblock_part2_5_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2ù
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_3781resblock_part2_6_conv1_3783*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_279020
.resblock_part2_6_conv1/StatefulPartitionedCall¸
&resblock_part2_6_relu1/PartitionedCallPartitionedCall7resblock_part2_6_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_28112(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_3787resblock_part2_6_conv2_3789*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_282920
.resblock_part2_6_conv2/StatefulPartitionedCallÄ
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x7resblock_part2_6_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2ù
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_3795resblock_part2_7_conv1_3797*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_285820
.resblock_part2_7_conv1/StatefulPartitionedCall¸
&resblock_part2_7_relu1/PartitionedCallPartitionedCall7resblock_part2_7_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_28792(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_3801resblock_part2_7_conv2_3803*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_289720
.resblock_part2_7_conv2/StatefulPartitionedCallÇ
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x7resblock_part2_7_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2ú
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_3809resblock_part2_8_conv1_3811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_292620
.resblock_part2_8_conv1/StatefulPartitionedCall¸
&resblock_part2_8_relu1/PartitionedCallPartitionedCall7resblock_part2_8_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_29472(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_3815resblock_part2_8_conv2_3817*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_296520
.resblock_part2_8_conv2/StatefulPartitionedCallÇ
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x7resblock_part2_8_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2Ä
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_3823upsampler_1_3825*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_1_layer_call_and_return_conditional_losses_29942%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_3829resblock_part3_1_conv1_3831*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_302120
.resblock_part3_1_conv1/StatefulPartitionedCallº
&resblock_part3_1_relu1/PartitionedCallPartitionedCall7resblock_part3_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_30422(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_3835resblock_part3_1_conv2_3837*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_306020
.resblock_part3_1_conv2/StatefulPartitionedCallÉ
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x7resblock_part3_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2ü
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_3843resblock_part3_2_conv1_3845*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_308920
.resblock_part3_2_conv1/StatefulPartitionedCallº
&resblock_part3_2_relu1/PartitionedCallPartitionedCall7resblock_part3_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_31102(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_3849resblock_part3_2_conv2_3851*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_312820
.resblock_part3_2_conv2/StatefulPartitionedCallÉ
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x7resblock_part3_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2ü
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_3857resblock_part3_3_conv1_3859*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_315720
.resblock_part3_3_conv1/StatefulPartitionedCallº
&resblock_part3_3_relu1/PartitionedCallPartitionedCall7resblock_part3_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_31782(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_3863resblock_part3_3_conv2_3865*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_319620
.resblock_part3_3_conv2/StatefulPartitionedCallÉ
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x7resblock_part3_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2ü
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_3871resblock_part3_4_conv1_3873*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_322520
.resblock_part3_4_conv1/StatefulPartitionedCallº
&resblock_part3_4_relu1/PartitionedCallPartitionedCall7resblock_part3_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_32462(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_3877resblock_part3_4_conv2_3879*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_326420
.resblock_part3_4_conv2/StatefulPartitionedCallÉ
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x7resblock_part3_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2À
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_3885extra_conv_3887*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_extra_conv_layer_call_and_return_conditional_losses_32932$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_3891upsampler_2_3893*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_2_layer_call_and_return_conditional_losses_33202%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_3897output_conv_3899*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_conv_layer_call_and_return_conditional_losses_33472%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
%downsampler_1/StatefulPartitionedCall%downsampler_1/StatefulPartitionedCall2N
%downsampler_2/StatefulPartitionedCall%downsampler_2/StatefulPartitionedCall2H
"extra_conv/StatefulPartitionedCall"extra_conv/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2`
.resblock_part1_1_conv1/StatefulPartitionedCall.resblock_part1_1_conv1/StatefulPartitionedCall2`
.resblock_part1_1_conv2/StatefulPartitionedCall.resblock_part1_1_conv2/StatefulPartitionedCall2`
.resblock_part1_2_conv1/StatefulPartitionedCall.resblock_part1_2_conv1/StatefulPartitionedCall2`
.resblock_part1_2_conv2/StatefulPartitionedCall.resblock_part1_2_conv2/StatefulPartitionedCall2`
.resblock_part1_3_conv1/StatefulPartitionedCall.resblock_part1_3_conv1/StatefulPartitionedCall2`
.resblock_part1_3_conv2/StatefulPartitionedCall.resblock_part1_3_conv2/StatefulPartitionedCall2`
.resblock_part1_4_conv1/StatefulPartitionedCall.resblock_part1_4_conv1/StatefulPartitionedCall2`
.resblock_part1_4_conv2/StatefulPartitionedCall.resblock_part1_4_conv2/StatefulPartitionedCall2`
.resblock_part2_1_conv1/StatefulPartitionedCall.resblock_part2_1_conv1/StatefulPartitionedCall2`
.resblock_part2_1_conv2/StatefulPartitionedCall.resblock_part2_1_conv2/StatefulPartitionedCall2`
.resblock_part2_2_conv1/StatefulPartitionedCall.resblock_part2_2_conv1/StatefulPartitionedCall2`
.resblock_part2_2_conv2/StatefulPartitionedCall.resblock_part2_2_conv2/StatefulPartitionedCall2`
.resblock_part2_3_conv1/StatefulPartitionedCall.resblock_part2_3_conv1/StatefulPartitionedCall2`
.resblock_part2_3_conv2/StatefulPartitionedCall.resblock_part2_3_conv2/StatefulPartitionedCall2`
.resblock_part2_4_conv1/StatefulPartitionedCall.resblock_part2_4_conv1/StatefulPartitionedCall2`
.resblock_part2_4_conv2/StatefulPartitionedCall.resblock_part2_4_conv2/StatefulPartitionedCall2`
.resblock_part2_5_conv1/StatefulPartitionedCall.resblock_part2_5_conv1/StatefulPartitionedCall2`
.resblock_part2_5_conv2/StatefulPartitionedCall.resblock_part2_5_conv2/StatefulPartitionedCall2`
.resblock_part2_6_conv1/StatefulPartitionedCall.resblock_part2_6_conv1/StatefulPartitionedCall2`
.resblock_part2_6_conv2/StatefulPartitionedCall.resblock_part2_6_conv2/StatefulPartitionedCall2`
.resblock_part2_7_conv1/StatefulPartitionedCall.resblock_part2_7_conv1/StatefulPartitionedCall2`
.resblock_part2_7_conv2/StatefulPartitionedCall.resblock_part2_7_conv2/StatefulPartitionedCall2`
.resblock_part2_8_conv1/StatefulPartitionedCall.resblock_part2_8_conv1/StatefulPartitionedCall2`
.resblock_part2_8_conv2/StatefulPartitionedCall.resblock_part2_8_conv2/StatefulPartitionedCall2`
.resblock_part3_1_conv1/StatefulPartitionedCall.resblock_part3_1_conv1/StatefulPartitionedCall2`
.resblock_part3_1_conv2/StatefulPartitionedCall.resblock_part3_1_conv2/StatefulPartitionedCall2`
.resblock_part3_2_conv1/StatefulPartitionedCall.resblock_part3_2_conv1/StatefulPartitionedCall2`
.resblock_part3_2_conv2/StatefulPartitionedCall.resblock_part3_2_conv2/StatefulPartitionedCall2`
.resblock_part3_3_conv1/StatefulPartitionedCall.resblock_part3_3_conv1/StatefulPartitionedCall2`
.resblock_part3_3_conv2/StatefulPartitionedCall.resblock_part3_3_conv2/StatefulPartitionedCall2`
.resblock_part3_4_conv1/StatefulPartitionedCall.resblock_part3_4_conv1/StatefulPartitionedCall2`
.resblock_part3_4_conv2/StatefulPartitionedCall.resblock_part3_4_conv2/StatefulPartitionedCall2J
#upsampler_1/StatefulPartitionedCall#upsampler_1/StatefulPartitionedCall2J
#upsampler_2/StatefulPartitionedCall#upsampler_2/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 


5__inference_resblock_part2_7_conv2_layer_call_fn_6339

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_28972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_2539

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_2219

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_1_conv2_layer_call_fn_6051

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_24892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_2761

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_1_conv2_layer_call_fn_5840

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_21902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

~
)__inference_input_conv_layer_call_fn_5773

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_input_conv_layer_call_and_return_conditional_losses_20982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_6560

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_5864

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®
K
/__inference_zero_padding2d_1_layer_call_fn_2084

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_20782
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

5__inference_resblock_part3_2_conv1_layer_call_fn_6473

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_30892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_3089

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_6478

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_5816

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_5831

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_4_conv2_layer_call_fn_5984

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_23942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨

Þ
E__inference_upsampler_2_layer_call_and_return_conditional_losses_3320

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp½
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp¢
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2	
BiasAdd 
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_3_conv1_layer_call_fn_5907

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_22872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
£

Þ
E__inference_output_conv_layer_call_and_return_conditional_losses_6646

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_2693

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_3246

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_5850

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_5912

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_5802

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_6363

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_5_relu1_layer_call_fn_6224

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_27432
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¦

à
G__inference_downsampler_1_layer_call_and_return_conditional_losses_2125

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp½
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢

Ý
D__inference_extra_conv_layer_call_and_return_conditional_losses_3293

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ä
f
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2078

inputs
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Pad
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

à
G__inference_downsampler_2_layer_call_and_return_conditional_losses_5994

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp»
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_2743

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_2811

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_1_conv1_layer_call_fn_5811

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_21512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_6061

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_2240

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_2722

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_6282

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_3_conv2_layer_call_fn_5936

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_23262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_2_conv2_layer_call_fn_6099

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_25572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_6512

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_2829

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_3196

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_6526

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_4_conv2_layer_call_fn_6598

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_32642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
îÆ
Á-
 __inference__traced_restore_7172
file_prefix&
"assignvariableop_input_conv_kernel&
"assignvariableop_1_input_conv_bias+
'assignvariableop_2_downsampler_1_kernel)
%assignvariableop_3_downsampler_1_bias4
0assignvariableop_4_resblock_part1_1_conv1_kernel2
.assignvariableop_5_resblock_part1_1_conv1_bias4
0assignvariableop_6_resblock_part1_1_conv2_kernel2
.assignvariableop_7_resblock_part1_1_conv2_bias4
0assignvariableop_8_resblock_part1_2_conv1_kernel2
.assignvariableop_9_resblock_part1_2_conv1_bias5
1assignvariableop_10_resblock_part1_2_conv2_kernel3
/assignvariableop_11_resblock_part1_2_conv2_bias5
1assignvariableop_12_resblock_part1_3_conv1_kernel3
/assignvariableop_13_resblock_part1_3_conv1_bias5
1assignvariableop_14_resblock_part1_3_conv2_kernel3
/assignvariableop_15_resblock_part1_3_conv2_bias5
1assignvariableop_16_resblock_part1_4_conv1_kernel3
/assignvariableop_17_resblock_part1_4_conv1_bias5
1assignvariableop_18_resblock_part1_4_conv2_kernel3
/assignvariableop_19_resblock_part1_4_conv2_bias,
(assignvariableop_20_downsampler_2_kernel*
&assignvariableop_21_downsampler_2_bias5
1assignvariableop_22_resblock_part2_1_conv1_kernel3
/assignvariableop_23_resblock_part2_1_conv1_bias5
1assignvariableop_24_resblock_part2_1_conv2_kernel3
/assignvariableop_25_resblock_part2_1_conv2_bias5
1assignvariableop_26_resblock_part2_2_conv1_kernel3
/assignvariableop_27_resblock_part2_2_conv1_bias5
1assignvariableop_28_resblock_part2_2_conv2_kernel3
/assignvariableop_29_resblock_part2_2_conv2_bias5
1assignvariableop_30_resblock_part2_3_conv1_kernel3
/assignvariableop_31_resblock_part2_3_conv1_bias5
1assignvariableop_32_resblock_part2_3_conv2_kernel3
/assignvariableop_33_resblock_part2_3_conv2_bias5
1assignvariableop_34_resblock_part2_4_conv1_kernel3
/assignvariableop_35_resblock_part2_4_conv1_bias5
1assignvariableop_36_resblock_part2_4_conv2_kernel3
/assignvariableop_37_resblock_part2_4_conv2_bias5
1assignvariableop_38_resblock_part2_5_conv1_kernel3
/assignvariableop_39_resblock_part2_5_conv1_bias5
1assignvariableop_40_resblock_part2_5_conv2_kernel3
/assignvariableop_41_resblock_part2_5_conv2_bias5
1assignvariableop_42_resblock_part2_6_conv1_kernel3
/assignvariableop_43_resblock_part2_6_conv1_bias5
1assignvariableop_44_resblock_part2_6_conv2_kernel3
/assignvariableop_45_resblock_part2_6_conv2_bias5
1assignvariableop_46_resblock_part2_7_conv1_kernel3
/assignvariableop_47_resblock_part2_7_conv1_bias5
1assignvariableop_48_resblock_part2_7_conv2_kernel3
/assignvariableop_49_resblock_part2_7_conv2_bias5
1assignvariableop_50_resblock_part2_8_conv1_kernel3
/assignvariableop_51_resblock_part2_8_conv1_bias5
1assignvariableop_52_resblock_part2_8_conv2_kernel3
/assignvariableop_53_resblock_part2_8_conv2_bias*
&assignvariableop_54_upsampler_1_kernel(
$assignvariableop_55_upsampler_1_bias5
1assignvariableop_56_resblock_part3_1_conv1_kernel3
/assignvariableop_57_resblock_part3_1_conv1_bias5
1assignvariableop_58_resblock_part3_1_conv2_kernel3
/assignvariableop_59_resblock_part3_1_conv2_bias5
1assignvariableop_60_resblock_part3_2_conv1_kernel3
/assignvariableop_61_resblock_part3_2_conv1_bias5
1assignvariableop_62_resblock_part3_2_conv2_kernel3
/assignvariableop_63_resblock_part3_2_conv2_bias5
1assignvariableop_64_resblock_part3_3_conv1_kernel3
/assignvariableop_65_resblock_part3_3_conv1_bias5
1assignvariableop_66_resblock_part3_3_conv2_kernel3
/assignvariableop_67_resblock_part3_3_conv2_bias5
1assignvariableop_68_resblock_part3_4_conv1_kernel3
/assignvariableop_69_resblock_part3_4_conv1_bias5
1assignvariableop_70_resblock_part3_4_conv2_kernel3
/assignvariableop_71_resblock_part3_4_conv2_bias)
%assignvariableop_72_extra_conv_kernel'
#assignvariableop_73_extra_conv_bias*
&assignvariableop_74_upsampler_2_kernel(
$assignvariableop_75_upsampler_2_bias*
&assignvariableop_76_output_conv_kernel(
$assignvariableop_77_output_conv_bias
identity_79¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_8¢AssignVariableOp_9£#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*¯"
value¥"B¢"OB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¯
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*³
value©B¦OB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¹
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ò
_output_shapes¿
¼:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*]
dtypesS
Q2O2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¡
AssignVariableOpAssignVariableOp"assignvariableop_input_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_input_conv_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¬
AssignVariableOp_2AssignVariableOp'assignvariableop_2_downsampler_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ª
AssignVariableOp_3AssignVariableOp%assignvariableop_3_downsampler_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4µ
AssignVariableOp_4AssignVariableOp0assignvariableop_4_resblock_part1_1_conv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_resblock_part1_1_conv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6µ
AssignVariableOp_6AssignVariableOp0assignvariableop_6_resblock_part1_1_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_resblock_part1_1_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8µ
AssignVariableOp_8AssignVariableOp0assignvariableop_8_resblock_part1_2_conv1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_resblock_part1_2_conv1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¹
AssignVariableOp_10AssignVariableOp1assignvariableop_10_resblock_part1_2_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11·
AssignVariableOp_11AssignVariableOp/assignvariableop_11_resblock_part1_2_conv2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¹
AssignVariableOp_12AssignVariableOp1assignvariableop_12_resblock_part1_3_conv1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13·
AssignVariableOp_13AssignVariableOp/assignvariableop_13_resblock_part1_3_conv1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¹
AssignVariableOp_14AssignVariableOp1assignvariableop_14_resblock_part1_3_conv2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15·
AssignVariableOp_15AssignVariableOp/assignvariableop_15_resblock_part1_3_conv2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¹
AssignVariableOp_16AssignVariableOp1assignvariableop_16_resblock_part1_4_conv1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17·
AssignVariableOp_17AssignVariableOp/assignvariableop_17_resblock_part1_4_conv1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¹
AssignVariableOp_18AssignVariableOp1assignvariableop_18_resblock_part1_4_conv2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19·
AssignVariableOp_19AssignVariableOp/assignvariableop_19_resblock_part1_4_conv2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_downsampler_2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp&assignvariableop_21_downsampler_2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¹
AssignVariableOp_22AssignVariableOp1assignvariableop_22_resblock_part2_1_conv1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23·
AssignVariableOp_23AssignVariableOp/assignvariableop_23_resblock_part2_1_conv1_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¹
AssignVariableOp_24AssignVariableOp1assignvariableop_24_resblock_part2_1_conv2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25·
AssignVariableOp_25AssignVariableOp/assignvariableop_25_resblock_part2_1_conv2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¹
AssignVariableOp_26AssignVariableOp1assignvariableop_26_resblock_part2_2_conv1_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27·
AssignVariableOp_27AssignVariableOp/assignvariableop_27_resblock_part2_2_conv1_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¹
AssignVariableOp_28AssignVariableOp1assignvariableop_28_resblock_part2_2_conv2_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29·
AssignVariableOp_29AssignVariableOp/assignvariableop_29_resblock_part2_2_conv2_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¹
AssignVariableOp_30AssignVariableOp1assignvariableop_30_resblock_part2_3_conv1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31·
AssignVariableOp_31AssignVariableOp/assignvariableop_31_resblock_part2_3_conv1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¹
AssignVariableOp_32AssignVariableOp1assignvariableop_32_resblock_part2_3_conv2_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33·
AssignVariableOp_33AssignVariableOp/assignvariableop_33_resblock_part2_3_conv2_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¹
AssignVariableOp_34AssignVariableOp1assignvariableop_34_resblock_part2_4_conv1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35·
AssignVariableOp_35AssignVariableOp/assignvariableop_35_resblock_part2_4_conv1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¹
AssignVariableOp_36AssignVariableOp1assignvariableop_36_resblock_part2_4_conv2_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37·
AssignVariableOp_37AssignVariableOp/assignvariableop_37_resblock_part2_4_conv2_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¹
AssignVariableOp_38AssignVariableOp1assignvariableop_38_resblock_part2_5_conv1_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39·
AssignVariableOp_39AssignVariableOp/assignvariableop_39_resblock_part2_5_conv1_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¹
AssignVariableOp_40AssignVariableOp1assignvariableop_40_resblock_part2_5_conv2_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41·
AssignVariableOp_41AssignVariableOp/assignvariableop_41_resblock_part2_5_conv2_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¹
AssignVariableOp_42AssignVariableOp1assignvariableop_42_resblock_part2_6_conv1_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43·
AssignVariableOp_43AssignVariableOp/assignvariableop_43_resblock_part2_6_conv1_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¹
AssignVariableOp_44AssignVariableOp1assignvariableop_44_resblock_part2_6_conv2_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45·
AssignVariableOp_45AssignVariableOp/assignvariableop_45_resblock_part2_6_conv2_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¹
AssignVariableOp_46AssignVariableOp1assignvariableop_46_resblock_part2_7_conv1_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47·
AssignVariableOp_47AssignVariableOp/assignvariableop_47_resblock_part2_7_conv1_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¹
AssignVariableOp_48AssignVariableOp1assignvariableop_48_resblock_part2_7_conv2_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49·
AssignVariableOp_49AssignVariableOp/assignvariableop_49_resblock_part2_7_conv2_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¹
AssignVariableOp_50AssignVariableOp1assignvariableop_50_resblock_part2_8_conv1_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51·
AssignVariableOp_51AssignVariableOp/assignvariableop_51_resblock_part2_8_conv1_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¹
AssignVariableOp_52AssignVariableOp1assignvariableop_52_resblock_part2_8_conv2_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53·
AssignVariableOp_53AssignVariableOp/assignvariableop_53_resblock_part2_8_conv2_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54®
AssignVariableOp_54AssignVariableOp&assignvariableop_54_upsampler_1_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¬
AssignVariableOp_55AssignVariableOp$assignvariableop_55_upsampler_1_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56¹
AssignVariableOp_56AssignVariableOp1assignvariableop_56_resblock_part3_1_conv1_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57·
AssignVariableOp_57AssignVariableOp/assignvariableop_57_resblock_part3_1_conv1_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58¹
AssignVariableOp_58AssignVariableOp1assignvariableop_58_resblock_part3_1_conv2_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59·
AssignVariableOp_59AssignVariableOp/assignvariableop_59_resblock_part3_1_conv2_biasIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60¹
AssignVariableOp_60AssignVariableOp1assignvariableop_60_resblock_part3_2_conv1_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61·
AssignVariableOp_61AssignVariableOp/assignvariableop_61_resblock_part3_2_conv1_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62¹
AssignVariableOp_62AssignVariableOp1assignvariableop_62_resblock_part3_2_conv2_kernelIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63·
AssignVariableOp_63AssignVariableOp/assignvariableop_63_resblock_part3_2_conv2_biasIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64¹
AssignVariableOp_64AssignVariableOp1assignvariableop_64_resblock_part3_3_conv1_kernelIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65·
AssignVariableOp_65AssignVariableOp/assignvariableop_65_resblock_part3_3_conv1_biasIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66¹
AssignVariableOp_66AssignVariableOp1assignvariableop_66_resblock_part3_3_conv2_kernelIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67·
AssignVariableOp_67AssignVariableOp/assignvariableop_67_resblock_part3_3_conv2_biasIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68¹
AssignVariableOp_68AssignVariableOp1assignvariableop_68_resblock_part3_4_conv1_kernelIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69·
AssignVariableOp_69AssignVariableOp/assignvariableop_69_resblock_part3_4_conv1_biasIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70¹
AssignVariableOp_70AssignVariableOp1assignvariableop_70_resblock_part3_4_conv2_kernelIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71·
AssignVariableOp_71AssignVariableOp/assignvariableop_71_resblock_part3_4_conv2_biasIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72­
AssignVariableOp_72AssignVariableOp%assignvariableop_72_extra_conv_kernelIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73«
AssignVariableOp_73AssignVariableOp#assignvariableop_73_extra_conv_biasIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74®
AssignVariableOp_74AssignVariableOp&assignvariableop_74_upsampler_2_kernelIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75¬
AssignVariableOp_75AssignVariableOp$assignvariableop_75_upsampler_2_biasIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76®
AssignVariableOp_76AssignVariableOp&assignvariableop_76_output_conv_kernelIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77¬
AssignVariableOp_77AssignVariableOp$assignvariableop_77_output_conv_biasIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_779
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_78Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_78
Identity_79IdentityIdentity_78:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_79"#
identity_79Identity_79:output:0*Ï
_input_shapes½
º: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
æ
l
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_3178

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_3_conv1_layer_call_fn_6118

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_25862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_2190

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_6205

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¢

Ý
D__inference_extra_conv_layer_call_and_return_conditional_losses_6608

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â
d
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_2065

inputs
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Pad
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_6234

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¢

Ý
D__inference_input_conv_layer_call_and_return_conditional_losses_2098

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_2394

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ª
I
-__inference_zero_padding2d_layer_call_fn_2071

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_20652
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_3021

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_6_conv1_layer_call_fn_6262

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_27902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_7_relu1_layer_call_fn_6320

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_28792
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_7_conv1_layer_call_fn_6310

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_28582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_3157

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_2_conv2_layer_call_fn_5888

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_22582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_1_conv1_layer_call_fn_6022

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_24502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part3_2_relu1_layer_call_fn_6483

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_31102
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_2376

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_2258

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_3225

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_2654

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part1_3_relu1_layer_call_fn_5917

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_23082
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_6574

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


*__inference_upsampler_1_layer_call_fn_6406

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_1_layer_call_and_return_conditional_losses_29942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_1_conv2_layer_call_fn_6454

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_30602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_2151

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part1_2_relu1_layer_call_fn_5869

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_22402
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¦

à
G__inference_downsampler_1_layer_call_and_return_conditional_losses_5783

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp½
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_6013

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_6075

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_2965

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


,__inference_downsampler_1_layer_call_fn_5792

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_1_layer_call_and_return_conditional_losses_21252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_2_relu1_layer_call_fn_6080

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_25392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_4_conv1_layer_call_fn_6166

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_26542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_3060

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_2790

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_2450

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
£

Þ
E__inference_output_conv_layer_call_and_return_conditional_losses_3347

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

5__inference_resblock_part1_4_conv1_layer_call_fn_5955

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_23552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_5879

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_2518

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_2926

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_6330

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

à
G__inference_downsampler_2_layer_call_and_return_conditional_losses_2424

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp»
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_6027

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_4_relu1_layer_call_fn_6176

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_26752
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_2607

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part1_4_relu1_layer_call_fn_5965

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_23762
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_2308

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_4_conv2_layer_call_fn_6195

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_26932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_5975

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_3042

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_3_relu1_layer_call_fn_6128

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_26072
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_6267

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_2_conv2_layer_call_fn_6502

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_31282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_6589

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ñ
í$
__inference__traced_save_6928
file_prefix0
,savev2_input_conv_kernel_read_readvariableop.
*savev2_input_conv_bias_read_readvariableop3
/savev2_downsampler_1_kernel_read_readvariableop1
-savev2_downsampler_1_bias_read_readvariableop<
8savev2_resblock_part1_1_conv1_kernel_read_readvariableop:
6savev2_resblock_part1_1_conv1_bias_read_readvariableop<
8savev2_resblock_part1_1_conv2_kernel_read_readvariableop:
6savev2_resblock_part1_1_conv2_bias_read_readvariableop<
8savev2_resblock_part1_2_conv1_kernel_read_readvariableop:
6savev2_resblock_part1_2_conv1_bias_read_readvariableop<
8savev2_resblock_part1_2_conv2_kernel_read_readvariableop:
6savev2_resblock_part1_2_conv2_bias_read_readvariableop<
8savev2_resblock_part1_3_conv1_kernel_read_readvariableop:
6savev2_resblock_part1_3_conv1_bias_read_readvariableop<
8savev2_resblock_part1_3_conv2_kernel_read_readvariableop:
6savev2_resblock_part1_3_conv2_bias_read_readvariableop<
8savev2_resblock_part1_4_conv1_kernel_read_readvariableop:
6savev2_resblock_part1_4_conv1_bias_read_readvariableop<
8savev2_resblock_part1_4_conv2_kernel_read_readvariableop:
6savev2_resblock_part1_4_conv2_bias_read_readvariableop3
/savev2_downsampler_2_kernel_read_readvariableop1
-savev2_downsampler_2_bias_read_readvariableop<
8savev2_resblock_part2_1_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_1_conv1_bias_read_readvariableop<
8savev2_resblock_part2_1_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_1_conv2_bias_read_readvariableop<
8savev2_resblock_part2_2_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_2_conv1_bias_read_readvariableop<
8savev2_resblock_part2_2_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_2_conv2_bias_read_readvariableop<
8savev2_resblock_part2_3_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_3_conv1_bias_read_readvariableop<
8savev2_resblock_part2_3_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_3_conv2_bias_read_readvariableop<
8savev2_resblock_part2_4_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_4_conv1_bias_read_readvariableop<
8savev2_resblock_part2_4_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_4_conv2_bias_read_readvariableop<
8savev2_resblock_part2_5_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_5_conv1_bias_read_readvariableop<
8savev2_resblock_part2_5_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_5_conv2_bias_read_readvariableop<
8savev2_resblock_part2_6_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_6_conv1_bias_read_readvariableop<
8savev2_resblock_part2_6_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_6_conv2_bias_read_readvariableop<
8savev2_resblock_part2_7_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_7_conv1_bias_read_readvariableop<
8savev2_resblock_part2_7_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_7_conv2_bias_read_readvariableop<
8savev2_resblock_part2_8_conv1_kernel_read_readvariableop:
6savev2_resblock_part2_8_conv1_bias_read_readvariableop<
8savev2_resblock_part2_8_conv2_kernel_read_readvariableop:
6savev2_resblock_part2_8_conv2_bias_read_readvariableop1
-savev2_upsampler_1_kernel_read_readvariableop/
+savev2_upsampler_1_bias_read_readvariableop<
8savev2_resblock_part3_1_conv1_kernel_read_readvariableop:
6savev2_resblock_part3_1_conv1_bias_read_readvariableop<
8savev2_resblock_part3_1_conv2_kernel_read_readvariableop:
6savev2_resblock_part3_1_conv2_bias_read_readvariableop<
8savev2_resblock_part3_2_conv1_kernel_read_readvariableop:
6savev2_resblock_part3_2_conv1_bias_read_readvariableop<
8savev2_resblock_part3_2_conv2_kernel_read_readvariableop:
6savev2_resblock_part3_2_conv2_bias_read_readvariableop<
8savev2_resblock_part3_3_conv1_kernel_read_readvariableop:
6savev2_resblock_part3_3_conv1_bias_read_readvariableop<
8savev2_resblock_part3_3_conv2_kernel_read_readvariableop:
6savev2_resblock_part3_3_conv2_bias_read_readvariableop<
8savev2_resblock_part3_4_conv1_kernel_read_readvariableop:
6savev2_resblock_part3_4_conv1_bias_read_readvariableop<
8savev2_resblock_part3_4_conv2_kernel_read_readvariableop:
6savev2_resblock_part3_4_conv2_bias_read_readvariableop0
,savev2_extra_conv_kernel_read_readvariableop.
*savev2_extra_conv_bias_read_readvariableop1
-savev2_upsampler_2_kernel_read_readvariableop/
+savev2_upsampler_2_bias_read_readvariableop1
-savev2_output_conv_kernel_read_readvariableop/
+savev2_output_conv_bias_read_readvariableop
savev2_const_16

identity_1¢MergeV2Checkpoints
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*¯"
value¥"B¢"OB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names©
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*³
value©B¦OB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÓ#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_input_conv_kernel_read_readvariableop*savev2_input_conv_bias_read_readvariableop/savev2_downsampler_1_kernel_read_readvariableop-savev2_downsampler_1_bias_read_readvariableop8savev2_resblock_part1_1_conv1_kernel_read_readvariableop6savev2_resblock_part1_1_conv1_bias_read_readvariableop8savev2_resblock_part1_1_conv2_kernel_read_readvariableop6savev2_resblock_part1_1_conv2_bias_read_readvariableop8savev2_resblock_part1_2_conv1_kernel_read_readvariableop6savev2_resblock_part1_2_conv1_bias_read_readvariableop8savev2_resblock_part1_2_conv2_kernel_read_readvariableop6savev2_resblock_part1_2_conv2_bias_read_readvariableop8savev2_resblock_part1_3_conv1_kernel_read_readvariableop6savev2_resblock_part1_3_conv1_bias_read_readvariableop8savev2_resblock_part1_3_conv2_kernel_read_readvariableop6savev2_resblock_part1_3_conv2_bias_read_readvariableop8savev2_resblock_part1_4_conv1_kernel_read_readvariableop6savev2_resblock_part1_4_conv1_bias_read_readvariableop8savev2_resblock_part1_4_conv2_kernel_read_readvariableop6savev2_resblock_part1_4_conv2_bias_read_readvariableop/savev2_downsampler_2_kernel_read_readvariableop-savev2_downsampler_2_bias_read_readvariableop8savev2_resblock_part2_1_conv1_kernel_read_readvariableop6savev2_resblock_part2_1_conv1_bias_read_readvariableop8savev2_resblock_part2_1_conv2_kernel_read_readvariableop6savev2_resblock_part2_1_conv2_bias_read_readvariableop8savev2_resblock_part2_2_conv1_kernel_read_readvariableop6savev2_resblock_part2_2_conv1_bias_read_readvariableop8savev2_resblock_part2_2_conv2_kernel_read_readvariableop6savev2_resblock_part2_2_conv2_bias_read_readvariableop8savev2_resblock_part2_3_conv1_kernel_read_readvariableop6savev2_resblock_part2_3_conv1_bias_read_readvariableop8savev2_resblock_part2_3_conv2_kernel_read_readvariableop6savev2_resblock_part2_3_conv2_bias_read_readvariableop8savev2_resblock_part2_4_conv1_kernel_read_readvariableop6savev2_resblock_part2_4_conv1_bias_read_readvariableop8savev2_resblock_part2_4_conv2_kernel_read_readvariableop6savev2_resblock_part2_4_conv2_bias_read_readvariableop8savev2_resblock_part2_5_conv1_kernel_read_readvariableop6savev2_resblock_part2_5_conv1_bias_read_readvariableop8savev2_resblock_part2_5_conv2_kernel_read_readvariableop6savev2_resblock_part2_5_conv2_bias_read_readvariableop8savev2_resblock_part2_6_conv1_kernel_read_readvariableop6savev2_resblock_part2_6_conv1_bias_read_readvariableop8savev2_resblock_part2_6_conv2_kernel_read_readvariableop6savev2_resblock_part2_6_conv2_bias_read_readvariableop8savev2_resblock_part2_7_conv1_kernel_read_readvariableop6savev2_resblock_part2_7_conv1_bias_read_readvariableop8savev2_resblock_part2_7_conv2_kernel_read_readvariableop6savev2_resblock_part2_7_conv2_bias_read_readvariableop8savev2_resblock_part2_8_conv1_kernel_read_readvariableop6savev2_resblock_part2_8_conv1_bias_read_readvariableop8savev2_resblock_part2_8_conv2_kernel_read_readvariableop6savev2_resblock_part2_8_conv2_bias_read_readvariableop-savev2_upsampler_1_kernel_read_readvariableop+savev2_upsampler_1_bias_read_readvariableop8savev2_resblock_part3_1_conv1_kernel_read_readvariableop6savev2_resblock_part3_1_conv1_bias_read_readvariableop8savev2_resblock_part3_1_conv2_kernel_read_readvariableop6savev2_resblock_part3_1_conv2_bias_read_readvariableop8savev2_resblock_part3_2_conv1_kernel_read_readvariableop6savev2_resblock_part3_2_conv1_bias_read_readvariableop8savev2_resblock_part3_2_conv2_kernel_read_readvariableop6savev2_resblock_part3_2_conv2_bias_read_readvariableop8savev2_resblock_part3_3_conv1_kernel_read_readvariableop6savev2_resblock_part3_3_conv1_bias_read_readvariableop8savev2_resblock_part3_3_conv2_kernel_read_readvariableop6savev2_resblock_part3_3_conv2_bias_read_readvariableop8savev2_resblock_part3_4_conv1_kernel_read_readvariableop6savev2_resblock_part3_4_conv1_bias_read_readvariableop8savev2_resblock_part3_4_conv2_kernel_read_readvariableop6savev2_resblock_part3_4_conv2_bias_read_readvariableop,savev2_extra_conv_kernel_read_readvariableop*savev2_extra_conv_bias_read_readvariableop-savev2_upsampler_2_kernel_read_readvariableop+savev2_upsampler_2_bias_read_readvariableop-savev2_output_conv_kernel_read_readvariableop+savev2_output_conv_bias_read_readvariableopsavev2_const_16"/device:CPU:0*
_output_shapes
 *]
dtypesS
Q2O2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Å
_input_shapes³
°: :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,	(
&
_output_shapes
:@@: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@:,!(
&
_output_shapes
:@@: "

_output_shapes
:@:,#(
&
_output_shapes
:@@: $

_output_shapes
:@:,%(
&
_output_shapes
:@@: &

_output_shapes
:@:,'(
&
_output_shapes
:@@: (

_output_shapes
:@:,)(
&
_output_shapes
:@@: *

_output_shapes
:@:,+(
&
_output_shapes
:@@: ,

_output_shapes
:@:,-(
&
_output_shapes
:@@: .

_output_shapes
:@:,/(
&
_output_shapes
:@@: 0

_output_shapes
:@:,1(
&
_output_shapes
:@@: 2

_output_shapes
:@:,3(
&
_output_shapes
:@@: 4

_output_shapes
:@:,5(
&
_output_shapes
:@@: 6

_output_shapes
:@:-7)
'
_output_shapes
:@:!8

_output_shapes	
::,9(
&
_output_shapes
:@@: :

_output_shapes
:@:,;(
&
_output_shapes
:@@: <

_output_shapes
:@:,=(
&
_output_shapes
:@@: >

_output_shapes
:@:,?(
&
_output_shapes
:@@: @

_output_shapes
:@:,A(
&
_output_shapes
:@@: B

_output_shapes
:@:,C(
&
_output_shapes
:@@: D

_output_shapes
:@:,E(
&
_output_shapes
:@@: F

_output_shapes
:@:,G(
&
_output_shapes
:@@: H

_output_shapes
:@:,I(
&
_output_shapes
:@@: J

_output_shapes
:@:-K)
'
_output_shapes
:@:!L

_output_shapes	
::,M(
&
_output_shapes
:@: N

_output_shapes
::O

_output_shapes
: 


*__inference_upsampler_2_layer_call_fn_6636

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_2_layer_call_and_return_conditional_losses_33202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_6541

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
©é
ò%
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_3632
input_layer
input_conv_3367
input_conv_3369
downsampler_1_3373
downsampler_1_3375
resblock_part1_1_conv1_3378
resblock_part1_1_conv1_3380
resblock_part1_1_conv2_3384
resblock_part1_1_conv2_3386
tf_math_multiply_mul_x
resblock_part1_2_conv1_3392
resblock_part1_2_conv1_3394
resblock_part1_2_conv2_3398
resblock_part1_2_conv2_3400
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_3406
resblock_part1_3_conv1_3408
resblock_part1_3_conv2_3412
resblock_part1_3_conv2_3414
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_3420
resblock_part1_4_conv1_3422
resblock_part1_4_conv2_3426
resblock_part1_4_conv2_3428
tf_math_multiply_3_mul_x
downsampler_2_3435
downsampler_2_3437
resblock_part2_1_conv1_3440
resblock_part2_1_conv1_3442
resblock_part2_1_conv2_3446
resblock_part2_1_conv2_3448
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_3454
resblock_part2_2_conv1_3456
resblock_part2_2_conv2_3460
resblock_part2_2_conv2_3462
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_3468
resblock_part2_3_conv1_3470
resblock_part2_3_conv2_3474
resblock_part2_3_conv2_3476
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_3482
resblock_part2_4_conv1_3484
resblock_part2_4_conv2_3488
resblock_part2_4_conv2_3490
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_3496
resblock_part2_5_conv1_3498
resblock_part2_5_conv2_3502
resblock_part2_5_conv2_3504
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_3510
resblock_part2_6_conv1_3512
resblock_part2_6_conv2_3516
resblock_part2_6_conv2_3518
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_3524
resblock_part2_7_conv1_3526
resblock_part2_7_conv2_3530
resblock_part2_7_conv2_3532
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_3538
resblock_part2_8_conv1_3540
resblock_part2_8_conv2_3544
resblock_part2_8_conv2_3546
tf_math_multiply_11_mul_x
upsampler_1_3552
upsampler_1_3554
resblock_part3_1_conv1_3558
resblock_part3_1_conv1_3560
resblock_part3_1_conv2_3564
resblock_part3_1_conv2_3566
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_3572
resblock_part3_2_conv1_3574
resblock_part3_2_conv2_3578
resblock_part3_2_conv2_3580
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_3586
resblock_part3_3_conv1_3588
resblock_part3_3_conv2_3592
resblock_part3_3_conv2_3594
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_3600
resblock_part3_4_conv1_3602
resblock_part3_4_conv2_3606
resblock_part3_4_conv2_3608
tf_math_multiply_15_mul_x
extra_conv_3614
extra_conv_3616
upsampler_2_3620
upsampler_2_3622
output_conv_3626
output_conv_3628
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCallª
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinput_layerinput_conv_3367input_conv_3369*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_input_conv_layer_call_and_return_conditional_losses_20982$
"input_conv/StatefulPartitionedCall
zero_padding2d/PartitionedCallPartitionedCall+input_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_20652 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_3373downsampler_1_3375*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_1_layer_call_and_return_conditional_losses_21252'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_3378resblock_part1_1_conv1_3380*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_215120
.resblock_part1_1_conv1/StatefulPartitionedCallº
&resblock_part1_1_relu1/PartitionedCallPartitionedCall7resblock_part1_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_21722(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_3384resblock_part1_1_conv2_3386*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_219020
.resblock_part1_1_conv2/StatefulPartitionedCallÀ
tf.math.multiply/MulMultf_math_multiply_mul_x7resblock_part1_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/MulÇ
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2ù
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_3392resblock_part1_2_conv1_3394*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_221920
.resblock_part1_2_conv1/StatefulPartitionedCallº
&resblock_part1_2_relu1/PartitionedCallPartitionedCall7resblock_part1_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_22402(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_3398resblock_part1_2_conv2_3400*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_225820
.resblock_part1_2_conv2/StatefulPartitionedCallÆ
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x7resblock_part1_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2û
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_3406resblock_part1_3_conv1_3408*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_228720
.resblock_part1_3_conv1/StatefulPartitionedCallº
&resblock_part1_3_relu1/PartitionedCallPartitionedCall7resblock_part1_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_23082(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_3412resblock_part1_3_conv2_3414*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_232620
.resblock_part1_3_conv2/StatefulPartitionedCallÆ
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x7resblock_part1_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2û
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_3420resblock_part1_4_conv1_3422*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_235520
.resblock_part1_4_conv1/StatefulPartitionedCallº
&resblock_part1_4_relu1/PartitionedCallPartitionedCall7resblock_part1_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_23762(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_3426resblock_part1_4_conv2_3428*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_239420
.resblock_part1_4_conv2/StatefulPartitionedCallÆ
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x7resblock_part1_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2
 zero_padding2d_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_20782"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_3435downsampler_2_3437*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_2_layer_call_and_return_conditional_losses_24242'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_3440resblock_part2_1_conv1_3442*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_245020
.resblock_part2_1_conv1/StatefulPartitionedCall¸
&resblock_part2_1_relu1/PartitionedCallPartitionedCall7resblock_part2_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_24712(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_3446resblock_part2_1_conv2_3448*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_248920
.resblock_part2_1_conv2/StatefulPartitionedCallÄ
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x7resblock_part2_1_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/MulË
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0.downsampler_2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2ù
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_3454resblock_part2_2_conv1_3456*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_251820
.resblock_part2_2_conv1/StatefulPartitionedCall¸
&resblock_part2_2_relu1/PartitionedCallPartitionedCall7resblock_part2_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_25392(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_3460resblock_part2_2_conv2_3462*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_255720
.resblock_part2_2_conv2/StatefulPartitionedCallÄ
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x7resblock_part2_2_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2ù
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_3468resblock_part2_3_conv1_3470*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_258620
.resblock_part2_3_conv1/StatefulPartitionedCall¸
&resblock_part2_3_relu1/PartitionedCallPartitionedCall7resblock_part2_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_26072(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_3474resblock_part2_3_conv2_3476*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_262520
.resblock_part2_3_conv2/StatefulPartitionedCallÄ
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x7resblock_part2_3_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2ù
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_3482resblock_part2_4_conv1_3484*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_265420
.resblock_part2_4_conv1/StatefulPartitionedCall¸
&resblock_part2_4_relu1/PartitionedCallPartitionedCall7resblock_part2_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_26752(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_3488resblock_part2_4_conv2_3490*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_269320
.resblock_part2_4_conv2/StatefulPartitionedCallÄ
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x7resblock_part2_4_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2ù
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_3496resblock_part2_5_conv1_3498*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_272220
.resblock_part2_5_conv1/StatefulPartitionedCall¸
&resblock_part2_5_relu1/PartitionedCallPartitionedCall7resblock_part2_5_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_27432(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_3502resblock_part2_5_conv2_3504*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_276120
.resblock_part2_5_conv2/StatefulPartitionedCallÄ
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x7resblock_part2_5_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2ù
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_3510resblock_part2_6_conv1_3512*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_279020
.resblock_part2_6_conv1/StatefulPartitionedCall¸
&resblock_part2_6_relu1/PartitionedCallPartitionedCall7resblock_part2_6_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_28112(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_3516resblock_part2_6_conv2_3518*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_282920
.resblock_part2_6_conv2/StatefulPartitionedCallÄ
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x7resblock_part2_6_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2ù
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_3524resblock_part2_7_conv1_3526*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_285820
.resblock_part2_7_conv1/StatefulPartitionedCall¸
&resblock_part2_7_relu1/PartitionedCallPartitionedCall7resblock_part2_7_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_28792(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_3530resblock_part2_7_conv2_3532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_289720
.resblock_part2_7_conv2/StatefulPartitionedCallÇ
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x7resblock_part2_7_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2ú
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_3538resblock_part2_8_conv1_3540*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_292620
.resblock_part2_8_conv1/StatefulPartitionedCall¸
&resblock_part2_8_relu1/PartitionedCallPartitionedCall7resblock_part2_8_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_29472(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_3544resblock_part2_8_conv2_3546*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_296520
.resblock_part2_8_conv2/StatefulPartitionedCallÇ
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x7resblock_part2_8_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2Ä
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_3552upsampler_1_3554*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_1_layer_call_and_return_conditional_losses_29942%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_3558resblock_part3_1_conv1_3560*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_302120
.resblock_part3_1_conv1/StatefulPartitionedCallº
&resblock_part3_1_relu1/PartitionedCallPartitionedCall7resblock_part3_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_30422(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_3564resblock_part3_1_conv2_3566*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_306020
.resblock_part3_1_conv2/StatefulPartitionedCallÉ
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x7resblock_part3_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2ü
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_3572resblock_part3_2_conv1_3574*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_308920
.resblock_part3_2_conv1/StatefulPartitionedCallº
&resblock_part3_2_relu1/PartitionedCallPartitionedCall7resblock_part3_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_31102(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_3578resblock_part3_2_conv2_3580*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_312820
.resblock_part3_2_conv2/StatefulPartitionedCallÉ
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x7resblock_part3_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2ü
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_3586resblock_part3_3_conv1_3588*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_315720
.resblock_part3_3_conv1/StatefulPartitionedCallº
&resblock_part3_3_relu1/PartitionedCallPartitionedCall7resblock_part3_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_31782(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_3592resblock_part3_3_conv2_3594*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_319620
.resblock_part3_3_conv2/StatefulPartitionedCallÉ
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x7resblock_part3_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2ü
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_3600resblock_part3_4_conv1_3602*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_322520
.resblock_part3_4_conv1/StatefulPartitionedCallº
&resblock_part3_4_relu1/PartitionedCallPartitionedCall7resblock_part3_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_32462(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_3606resblock_part3_4_conv2_3608*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_326420
.resblock_part3_4_conv2/StatefulPartitionedCallÉ
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x7resblock_part3_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2À
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_3614extra_conv_3616*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_extra_conv_layer_call_and_return_conditional_losses_32932$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_3620upsampler_2_3622*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_2_layer_call_and_return_conditional_losses_33202%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_3626output_conv_3628*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_conv_layer_call_and_return_conditional_losses_33472%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
%downsampler_1/StatefulPartitionedCall%downsampler_1/StatefulPartitionedCall2N
%downsampler_2/StatefulPartitionedCall%downsampler_2/StatefulPartitionedCall2H
"extra_conv/StatefulPartitionedCall"extra_conv/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2`
.resblock_part1_1_conv1/StatefulPartitionedCall.resblock_part1_1_conv1/StatefulPartitionedCall2`
.resblock_part1_1_conv2/StatefulPartitionedCall.resblock_part1_1_conv2/StatefulPartitionedCall2`
.resblock_part1_2_conv1/StatefulPartitionedCall.resblock_part1_2_conv1/StatefulPartitionedCall2`
.resblock_part1_2_conv2/StatefulPartitionedCall.resblock_part1_2_conv2/StatefulPartitionedCall2`
.resblock_part1_3_conv1/StatefulPartitionedCall.resblock_part1_3_conv1/StatefulPartitionedCall2`
.resblock_part1_3_conv2/StatefulPartitionedCall.resblock_part1_3_conv2/StatefulPartitionedCall2`
.resblock_part1_4_conv1/StatefulPartitionedCall.resblock_part1_4_conv1/StatefulPartitionedCall2`
.resblock_part1_4_conv2/StatefulPartitionedCall.resblock_part1_4_conv2/StatefulPartitionedCall2`
.resblock_part2_1_conv1/StatefulPartitionedCall.resblock_part2_1_conv1/StatefulPartitionedCall2`
.resblock_part2_1_conv2/StatefulPartitionedCall.resblock_part2_1_conv2/StatefulPartitionedCall2`
.resblock_part2_2_conv1/StatefulPartitionedCall.resblock_part2_2_conv1/StatefulPartitionedCall2`
.resblock_part2_2_conv2/StatefulPartitionedCall.resblock_part2_2_conv2/StatefulPartitionedCall2`
.resblock_part2_3_conv1/StatefulPartitionedCall.resblock_part2_3_conv1/StatefulPartitionedCall2`
.resblock_part2_3_conv2/StatefulPartitionedCall.resblock_part2_3_conv2/StatefulPartitionedCall2`
.resblock_part2_4_conv1/StatefulPartitionedCall.resblock_part2_4_conv1/StatefulPartitionedCall2`
.resblock_part2_4_conv2/StatefulPartitionedCall.resblock_part2_4_conv2/StatefulPartitionedCall2`
.resblock_part2_5_conv1/StatefulPartitionedCall.resblock_part2_5_conv1/StatefulPartitionedCall2`
.resblock_part2_5_conv2/StatefulPartitionedCall.resblock_part2_5_conv2/StatefulPartitionedCall2`
.resblock_part2_6_conv1/StatefulPartitionedCall.resblock_part2_6_conv1/StatefulPartitionedCall2`
.resblock_part2_6_conv2/StatefulPartitionedCall.resblock_part2_6_conv2/StatefulPartitionedCall2`
.resblock_part2_7_conv1/StatefulPartitionedCall.resblock_part2_7_conv1/StatefulPartitionedCall2`
.resblock_part2_7_conv2/StatefulPartitionedCall.resblock_part2_7_conv2/StatefulPartitionedCall2`
.resblock_part2_8_conv1/StatefulPartitionedCall.resblock_part2_8_conv1/StatefulPartitionedCall2`
.resblock_part2_8_conv2/StatefulPartitionedCall.resblock_part2_8_conv2/StatefulPartitionedCall2`
.resblock_part3_1_conv1/StatefulPartitionedCall.resblock_part3_1_conv1/StatefulPartitionedCall2`
.resblock_part3_1_conv2/StatefulPartitionedCall.resblock_part3_1_conv2/StatefulPartitionedCall2`
.resblock_part3_2_conv1/StatefulPartitionedCall.resblock_part3_2_conv1/StatefulPartitionedCall2`
.resblock_part3_2_conv2/StatefulPartitionedCall.resblock_part3_2_conv2/StatefulPartitionedCall2`
.resblock_part3_3_conv1/StatefulPartitionedCall.resblock_part3_3_conv1/StatefulPartitionedCall2`
.resblock_part3_3_conv2/StatefulPartitionedCall.resblock_part3_3_conv2/StatefulPartitionedCall2`
.resblock_part3_4_conv1/StatefulPartitionedCall.resblock_part3_4_conv1/StatefulPartitionedCall2`
.resblock_part3_4_conv2/StatefulPartitionedCall.resblock_part3_4_conv2/StatefulPartitionedCall2J
#upsampler_1/StatefulPartitionedCall#upsampler_1/StatefulPartitionedCall2J
#upsampler_2/StatefulPartitionedCall#upsampler_2/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
Þ
l
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_2947

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part1_1_relu1_layer_call_fn_5821

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_21722
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ô"
¼
+__inference_ssi_res_unet_layer_call_fn_4555
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_43642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
®

é
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_2326

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part3_4_relu1_layer_call_fn_6579

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_32462
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_3_conv2_layer_call_fn_6147

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_26252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_5898

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢

Ý
D__inference_input_conv_layer_call_and_return_conditional_losses_5764

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_6138

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_5_conv2_layer_call_fn_6243

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_27612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_6042

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_3110

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_2489

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_3128

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

~
)__inference_extra_conv_layer_call_fn_6617

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_extra_conv_layer_call_and_return_conditional_losses_32932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_5960

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_5946

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ä"
³
"__inference_signature_wrapper_4750
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity¢StatefulPartitionedCallÐ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_20582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
Í
Q
5__inference_resblock_part2_8_relu1_layer_call_fn_6368

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_29472
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_2557

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_6090

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_1_relu1_layer_call_fn_6032

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_24712
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_6186

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_6171

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

5__inference_resblock_part3_1_conv1_layer_call_fn_6425

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_30212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_6416

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_3264

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
é
í%
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4364

inputs
input_conv_4099
input_conv_4101
downsampler_1_4105
downsampler_1_4107
resblock_part1_1_conv1_4110
resblock_part1_1_conv1_4112
resblock_part1_1_conv2_4116
resblock_part1_1_conv2_4118
tf_math_multiply_mul_x
resblock_part1_2_conv1_4124
resblock_part1_2_conv1_4126
resblock_part1_2_conv2_4130
resblock_part1_2_conv2_4132
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_4138
resblock_part1_3_conv1_4140
resblock_part1_3_conv2_4144
resblock_part1_3_conv2_4146
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_4152
resblock_part1_4_conv1_4154
resblock_part1_4_conv2_4158
resblock_part1_4_conv2_4160
tf_math_multiply_3_mul_x
downsampler_2_4167
downsampler_2_4169
resblock_part2_1_conv1_4172
resblock_part2_1_conv1_4174
resblock_part2_1_conv2_4178
resblock_part2_1_conv2_4180
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_4186
resblock_part2_2_conv1_4188
resblock_part2_2_conv2_4192
resblock_part2_2_conv2_4194
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_4200
resblock_part2_3_conv1_4202
resblock_part2_3_conv2_4206
resblock_part2_3_conv2_4208
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_4214
resblock_part2_4_conv1_4216
resblock_part2_4_conv2_4220
resblock_part2_4_conv2_4222
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_4228
resblock_part2_5_conv1_4230
resblock_part2_5_conv2_4234
resblock_part2_5_conv2_4236
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_4242
resblock_part2_6_conv1_4244
resblock_part2_6_conv2_4248
resblock_part2_6_conv2_4250
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_4256
resblock_part2_7_conv1_4258
resblock_part2_7_conv2_4262
resblock_part2_7_conv2_4264
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_4270
resblock_part2_8_conv1_4272
resblock_part2_8_conv2_4276
resblock_part2_8_conv2_4278
tf_math_multiply_11_mul_x
upsampler_1_4284
upsampler_1_4286
resblock_part3_1_conv1_4290
resblock_part3_1_conv1_4292
resblock_part3_1_conv2_4296
resblock_part3_1_conv2_4298
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_4304
resblock_part3_2_conv1_4306
resblock_part3_2_conv2_4310
resblock_part3_2_conv2_4312
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_4318
resblock_part3_3_conv1_4320
resblock_part3_3_conv2_4324
resblock_part3_3_conv2_4326
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_4332
resblock_part3_4_conv1_4334
resblock_part3_4_conv2_4338
resblock_part3_4_conv2_4340
tf_math_multiply_15_mul_x
extra_conv_4346
extra_conv_4348
upsampler_2_4352
upsampler_2_4354
output_conv_4358
output_conv_4360
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCall¥
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_4099input_conv_4101*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_input_conv_layer_call_and_return_conditional_losses_20982$
"input_conv/StatefulPartitionedCall
zero_padding2d/PartitionedCallPartitionedCall+input_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_20652 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_4105downsampler_1_4107*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_1_layer_call_and_return_conditional_losses_21252'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_4110resblock_part1_1_conv1_4112*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_215120
.resblock_part1_1_conv1/StatefulPartitionedCallº
&resblock_part1_1_relu1/PartitionedCallPartitionedCall7resblock_part1_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_21722(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_4116resblock_part1_1_conv2_4118*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_219020
.resblock_part1_1_conv2/StatefulPartitionedCallÀ
tf.math.multiply/MulMultf_math_multiply_mul_x7resblock_part1_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply/MulÇ
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add/AddV2ù
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_4124resblock_part1_2_conv1_4126*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_221920
.resblock_part1_2_conv1/StatefulPartitionedCallº
&resblock_part1_2_relu1/PartitionedCallPartitionedCall7resblock_part1_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_22402(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_4130resblock_part1_2_conv2_4132*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_225820
.resblock_part1_2_conv2/StatefulPartitionedCallÆ
tf.math.multiply_1/MulMultf_math_multiply_1_mul_x7resblock_part1_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_1/Mul½
tf.__operators__.add_1/AddV2AddV2tf.math.multiply_1/Mul:z:0tf.__operators__.add/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_1/AddV2û
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_4138resblock_part1_3_conv1_4140*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_228720
.resblock_part1_3_conv1/StatefulPartitionedCallº
&resblock_part1_3_relu1/PartitionedCallPartitionedCall7resblock_part1_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_23082(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_4144resblock_part1_3_conv2_4146*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_232620
.resblock_part1_3_conv2/StatefulPartitionedCallÆ
tf.math.multiply_2/MulMultf_math_multiply_2_mul_x7resblock_part1_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_2/Mul¿
tf.__operators__.add_2/AddV2AddV2tf.math.multiply_2/Mul:z:0 tf.__operators__.add_1/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_2/AddV2û
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_4152resblock_part1_4_conv1_4154*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_235520
.resblock_part1_4_conv1/StatefulPartitionedCallº
&resblock_part1_4_relu1/PartitionedCallPartitionedCall7resblock_part1_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_23762(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_4158resblock_part1_4_conv2_4160*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_239420
.resblock_part1_4_conv2/StatefulPartitionedCallÆ
tf.math.multiply_3/MulMultf_math_multiply_3_mul_x7resblock_part1_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_3/Mul¿
tf.__operators__.add_3/AddV2AddV2tf.math.multiply_3/Mul:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_3/AddV2
 zero_padding2d_1/PartitionedCallPartitionedCall tf.__operators__.add_3/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_20782"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_4167downsampler_2_4169*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_2_layer_call_and_return_conditional_losses_24242'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_4172resblock_part2_1_conv1_4174*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_245020
.resblock_part2_1_conv1/StatefulPartitionedCall¸
&resblock_part2_1_relu1/PartitionedCallPartitionedCall7resblock_part2_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_24712(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_4178resblock_part2_1_conv2_4180*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_248920
.resblock_part2_1_conv2/StatefulPartitionedCallÄ
tf.math.multiply_4/MulMultf_math_multiply_4_mul_x7resblock_part2_1_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_4/MulË
tf.__operators__.add_4/AddV2AddV2tf.math.multiply_4/Mul:z:0.downsampler_2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_4/AddV2ù
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_4186resblock_part2_2_conv1_4188*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_251820
.resblock_part2_2_conv1/StatefulPartitionedCall¸
&resblock_part2_2_relu1/PartitionedCallPartitionedCall7resblock_part2_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_25392(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_4192resblock_part2_2_conv2_4194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_255720
.resblock_part2_2_conv2/StatefulPartitionedCallÄ
tf.math.multiply_5/MulMultf_math_multiply_5_mul_x7resblock_part2_2_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_5/Mul½
tf.__operators__.add_5/AddV2AddV2tf.math.multiply_5/Mul:z:0 tf.__operators__.add_4/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_5/AddV2ù
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_4200resblock_part2_3_conv1_4202*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_258620
.resblock_part2_3_conv1/StatefulPartitionedCall¸
&resblock_part2_3_relu1/PartitionedCallPartitionedCall7resblock_part2_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_26072(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_4206resblock_part2_3_conv2_4208*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_262520
.resblock_part2_3_conv2/StatefulPartitionedCallÄ
tf.math.multiply_6/MulMultf_math_multiply_6_mul_x7resblock_part2_3_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_6/Mul½
tf.__operators__.add_6/AddV2AddV2tf.math.multiply_6/Mul:z:0 tf.__operators__.add_5/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_6/AddV2ù
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_4214resblock_part2_4_conv1_4216*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_265420
.resblock_part2_4_conv1/StatefulPartitionedCall¸
&resblock_part2_4_relu1/PartitionedCallPartitionedCall7resblock_part2_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_26752(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_4220resblock_part2_4_conv2_4222*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_269320
.resblock_part2_4_conv2/StatefulPartitionedCallÄ
tf.math.multiply_7/MulMultf_math_multiply_7_mul_x7resblock_part2_4_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_7/Mul½
tf.__operators__.add_7/AddV2AddV2tf.math.multiply_7/Mul:z:0 tf.__operators__.add_6/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_7/AddV2ù
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_4228resblock_part2_5_conv1_4230*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_272220
.resblock_part2_5_conv1/StatefulPartitionedCall¸
&resblock_part2_5_relu1/PartitionedCallPartitionedCall7resblock_part2_5_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_27432(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_4234resblock_part2_5_conv2_4236*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_276120
.resblock_part2_5_conv2/StatefulPartitionedCallÄ
tf.math.multiply_8/MulMultf_math_multiply_8_mul_x7resblock_part2_5_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_8/Mul½
tf.__operators__.add_8/AddV2AddV2tf.math.multiply_8/Mul:z:0 tf.__operators__.add_7/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_8/AddV2ù
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_4242resblock_part2_6_conv1_4244*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_279020
.resblock_part2_6_conv1/StatefulPartitionedCall¸
&resblock_part2_6_relu1/PartitionedCallPartitionedCall7resblock_part2_6_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_28112(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_4248resblock_part2_6_conv2_4250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_282920
.resblock_part2_6_conv2/StatefulPartitionedCallÄ
tf.math.multiply_9/MulMultf_math_multiply_9_mul_x7resblock_part2_6_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_9/Mul½
tf.__operators__.add_9/AddV2AddV2tf.math.multiply_9/Mul:z:0 tf.__operators__.add_8/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_9/AddV2ù
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_4256resblock_part2_7_conv1_4258*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_285820
.resblock_part2_7_conv1/StatefulPartitionedCall¸
&resblock_part2_7_relu1/PartitionedCallPartitionedCall7resblock_part2_7_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_28792(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_4262resblock_part2_7_conv2_4264*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_289720
.resblock_part2_7_conv2/StatefulPartitionedCallÇ
tf.math.multiply_10/MulMultf_math_multiply_10_mul_x7resblock_part2_7_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_10/MulÀ
tf.__operators__.add_10/AddV2AddV2tf.math.multiply_10/Mul:z:0 tf.__operators__.add_9/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_10/AddV2ú
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_4270resblock_part2_8_conv1_4272*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_292620
.resblock_part2_8_conv1/StatefulPartitionedCall¸
&resblock_part2_8_relu1/PartitionedCallPartitionedCall7resblock_part2_8_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_29472(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_4276resblock_part2_8_conv2_4278*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_296520
.resblock_part2_8_conv2/StatefulPartitionedCallÇ
tf.math.multiply_11/MulMultf_math_multiply_11_mul_x7resblock_part2_8_conv2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.math.multiply_11/MulÁ
tf.__operators__.add_11/AddV2AddV2tf.math.multiply_11/Mul:z:0!tf.__operators__.add_10/AddV2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
tf.__operators__.add_11/AddV2Ä
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_4284upsampler_1_4286*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_1_layer_call_and_return_conditional_losses_29942%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_4290resblock_part3_1_conv1_4292*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_302120
.resblock_part3_1_conv1/StatefulPartitionedCallº
&resblock_part3_1_relu1/PartitionedCallPartitionedCall7resblock_part3_1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_30422(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_4296resblock_part3_1_conv2_4298*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_306020
.resblock_part3_1_conv2/StatefulPartitionedCallÉ
tf.math.multiply_12/MulMultf_math_multiply_12_mul_x7resblock_part3_1_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_12/MulÌ
tf.__operators__.add_12/AddV2AddV2tf.math.multiply_12/Mul:z:0*tf.nn.depth_to_space/DepthToSpace:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_12/AddV2ü
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_4304resblock_part3_2_conv1_4306*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_308920
.resblock_part3_2_conv1/StatefulPartitionedCallº
&resblock_part3_2_relu1/PartitionedCallPartitionedCall7resblock_part3_2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_31102(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_4310resblock_part3_2_conv2_4312*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_312820
.resblock_part3_2_conv2/StatefulPartitionedCallÉ
tf.math.multiply_13/MulMultf_math_multiply_13_mul_x7resblock_part3_2_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_13/MulÃ
tf.__operators__.add_13/AddV2AddV2tf.math.multiply_13/Mul:z:0!tf.__operators__.add_12/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_13/AddV2ü
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_4318resblock_part3_3_conv1_4320*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_315720
.resblock_part3_3_conv1/StatefulPartitionedCallº
&resblock_part3_3_relu1/PartitionedCallPartitionedCall7resblock_part3_3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_31782(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_4324resblock_part3_3_conv2_4326*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_319620
.resblock_part3_3_conv2/StatefulPartitionedCallÉ
tf.math.multiply_14/MulMultf_math_multiply_14_mul_x7resblock_part3_3_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_14/MulÃ
tf.__operators__.add_14/AddV2AddV2tf.math.multiply_14/Mul:z:0!tf.__operators__.add_13/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_14/AddV2ü
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_4332resblock_part3_4_conv1_4334*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_322520
.resblock_part3_4_conv1/StatefulPartitionedCallº
&resblock_part3_4_relu1/PartitionedCallPartitionedCall7resblock_part3_4_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_32462(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_4338resblock_part3_4_conv2_4340*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_326420
.resblock_part3_4_conv2/StatefulPartitionedCallÉ
tf.math.multiply_15/MulMultf_math_multiply_15_mul_x7resblock_part3_4_conv2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.math.multiply_15/MulÃ
tf.__operators__.add_15/AddV2AddV2tf.math.multiply_15/Mul:z:0!tf.__operators__.add_14/AddV2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_15/AddV2À
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_4346extra_conv_4348*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_extra_conv_layer_call_and_return_conditional_losses_32932$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_4352upsampler_2_4354*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_upsampler_2_layer_call_and_return_conditional_losses_33202%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_4358output_conv_4360*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_conv_layer_call_and_return_conditional_losses_33472%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
%downsampler_1/StatefulPartitionedCall%downsampler_1/StatefulPartitionedCall2N
%downsampler_2/StatefulPartitionedCall%downsampler_2/StatefulPartitionedCall2H
"extra_conv/StatefulPartitionedCall"extra_conv/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2`
.resblock_part1_1_conv1/StatefulPartitionedCall.resblock_part1_1_conv1/StatefulPartitionedCall2`
.resblock_part1_1_conv2/StatefulPartitionedCall.resblock_part1_1_conv2/StatefulPartitionedCall2`
.resblock_part1_2_conv1/StatefulPartitionedCall.resblock_part1_2_conv1/StatefulPartitionedCall2`
.resblock_part1_2_conv2/StatefulPartitionedCall.resblock_part1_2_conv2/StatefulPartitionedCall2`
.resblock_part1_3_conv1/StatefulPartitionedCall.resblock_part1_3_conv1/StatefulPartitionedCall2`
.resblock_part1_3_conv2/StatefulPartitionedCall.resblock_part1_3_conv2/StatefulPartitionedCall2`
.resblock_part1_4_conv1/StatefulPartitionedCall.resblock_part1_4_conv1/StatefulPartitionedCall2`
.resblock_part1_4_conv2/StatefulPartitionedCall.resblock_part1_4_conv2/StatefulPartitionedCall2`
.resblock_part2_1_conv1/StatefulPartitionedCall.resblock_part2_1_conv1/StatefulPartitionedCall2`
.resblock_part2_1_conv2/StatefulPartitionedCall.resblock_part2_1_conv2/StatefulPartitionedCall2`
.resblock_part2_2_conv1/StatefulPartitionedCall.resblock_part2_2_conv1/StatefulPartitionedCall2`
.resblock_part2_2_conv2/StatefulPartitionedCall.resblock_part2_2_conv2/StatefulPartitionedCall2`
.resblock_part2_3_conv1/StatefulPartitionedCall.resblock_part2_3_conv1/StatefulPartitionedCall2`
.resblock_part2_3_conv2/StatefulPartitionedCall.resblock_part2_3_conv2/StatefulPartitionedCall2`
.resblock_part2_4_conv1/StatefulPartitionedCall.resblock_part2_4_conv1/StatefulPartitionedCall2`
.resblock_part2_4_conv2/StatefulPartitionedCall.resblock_part2_4_conv2/StatefulPartitionedCall2`
.resblock_part2_5_conv1/StatefulPartitionedCall.resblock_part2_5_conv1/StatefulPartitionedCall2`
.resblock_part2_5_conv2/StatefulPartitionedCall.resblock_part2_5_conv2/StatefulPartitionedCall2`
.resblock_part2_6_conv1/StatefulPartitionedCall.resblock_part2_6_conv1/StatefulPartitionedCall2`
.resblock_part2_6_conv2/StatefulPartitionedCall.resblock_part2_6_conv2/StatefulPartitionedCall2`
.resblock_part2_7_conv1/StatefulPartitionedCall.resblock_part2_7_conv1/StatefulPartitionedCall2`
.resblock_part2_7_conv2/StatefulPartitionedCall.resblock_part2_7_conv2/StatefulPartitionedCall2`
.resblock_part2_8_conv1/StatefulPartitionedCall.resblock_part2_8_conv1/StatefulPartitionedCall2`
.resblock_part2_8_conv2/StatefulPartitionedCall.resblock_part2_8_conv2/StatefulPartitionedCall2`
.resblock_part3_1_conv1/StatefulPartitionedCall.resblock_part3_1_conv1/StatefulPartitionedCall2`
.resblock_part3_1_conv2/StatefulPartitionedCall.resblock_part3_1_conv2/StatefulPartitionedCall2`
.resblock_part3_2_conv1/StatefulPartitionedCall.resblock_part3_2_conv1/StatefulPartitionedCall2`
.resblock_part3_2_conv2/StatefulPartitionedCall.resblock_part3_2_conv2/StatefulPartitionedCall2`
.resblock_part3_3_conv1/StatefulPartitionedCall.resblock_part3_3_conv1/StatefulPartitionedCall2`
.resblock_part3_3_conv2/StatefulPartitionedCall.resblock_part3_3_conv2/StatefulPartitionedCall2`
.resblock_part3_4_conv1/StatefulPartitionedCall.resblock_part3_4_conv1/StatefulPartitionedCall2`
.resblock_part3_4_conv2/StatefulPartitionedCall.resblock_part3_4_conv2/StatefulPartitionedCall2J
#upsampler_1/StatefulPartitionedCall#upsampler_1/StatefulPartitionedCall2J
#upsampler_2/StatefulPartitionedCall#upsampler_2/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
 

5__inference_resblock_part1_2_conv1_layer_call_fn_5859

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_22192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_5927

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_6315

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


5__inference_resblock_part2_8_conv2_layer_call_fn_6387

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_29652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_6109

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_2355

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_2879

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_2586

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_2858

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_6123

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
æ
l
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_2172

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_6349

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_2897

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part3_3_relu1_layer_call_fn_6531

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_31782
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ
Q
5__inference_resblock_part3_1_relu1_layer_call_fn_6435

inputs
identityÛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_30422
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
å"
·
+__inference_ssi_res_unet_layer_call_fn_5561

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_39032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 


Þ
E__inference_upsampler_1_layer_call_and_return_conditional_losses_6397

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp»
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp 
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Þ
l
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_2675

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
å"
·
+__inference_ssi_res_unet_layer_call_fn_5754

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84

unknown_85

unknown_86

unknown_87

unknown_88

unknown_89

unknown_90

unknown_91

unknown_92
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92*j
Tinc
a2_*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*p
_read_only_resource_inputsR
PN
 !"#%&'(*+,-/01245679:;<>?@ACDEFGHJKLMOPQRTUVWYZ[\]^*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_43642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:	

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$

_output_shapes
: :)

_output_shapes
: :.

_output_shapes
: :3

_output_shapes
: :8

_output_shapes
: :=

_output_shapes
: :B

_output_shapes
: :I

_output_shapes
: :N

_output_shapes
: :S

_output_shapes
: :X

_output_shapes
: 
æ
l
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_6430

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_6301

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpº
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


,__inference_downsampler_2_layer_call_fn_6003

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_downsampler_2_layer_call_and_return_conditional_losses_24242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


5__inference_resblock_part2_2_conv1_layer_call_fn_6070

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_25182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs


*__inference_output_conv_layer_call_fn_6655

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_conv_layer_call_and_return_conditional_losses_33472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨

Þ
E__inference_upsampler_2_layer_call_and_return_conditional_losses_6627

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp½
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp¢
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ*
data_formatNCHW2	
BiasAdd 
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®

é
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_2287

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¼
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¡
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
data_formatNCHW2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
Q
5__inference_resblock_part2_6_relu1_layer_call_fn_6272

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_28112
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ê
serving_default¶
M
input_layer>
serving_default_input_layer:0ÿÿÿÿÿÿÿÿÿI
output_conv:
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:è
Þ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer-24
layer_with_weights-10
layer-25
layer_with_weights-11
layer-26
layer-27
layer_with_weights-12
layer-28
layer-29
layer-30
 layer_with_weights-13
 layer-31
!layer-32
"layer_with_weights-14
"layer-33
#layer-34
$layer-35
%layer_with_weights-15
%layer-36
&layer-37
'layer_with_weights-16
'layer-38
(layer-39
)layer-40
*layer_with_weights-17
*layer-41
+layer-42
,layer_with_weights-18
,layer-43
-layer-44
.layer-45
/layer_with_weights-19
/layer-46
0layer-47
1layer_with_weights-20
1layer-48
2layer-49
3layer-50
4layer_with_weights-21
4layer-51
5layer-52
6layer_with_weights-22
6layer-53
7layer-54
8layer-55
9layer_with_weights-23
9layer-56
:layer-57
;layer_with_weights-24
;layer-58
<layer-59
=layer-60
>layer_with_weights-25
>layer-61
?layer-62
@layer_with_weights-26
@layer-63
Alayer-64
Blayer-65
Clayer_with_weights-27
Clayer-66
Dlayer-67
Elayer_with_weights-28
Elayer-68
Flayer-69
Glayer_with_weights-29
Glayer-70
Hlayer-71
Ilayer-72
Jlayer_with_weights-30
Jlayer-73
Klayer-74
Llayer_with_weights-31
Llayer-75
Mlayer-76
Nlayer-77
Olayer_with_weights-32
Olayer-78
Player-79
Qlayer_with_weights-33
Qlayer-80
Rlayer-81
Slayer-82
Tlayer_with_weights-34
Tlayer-83
Ulayer-84
Vlayer_with_weights-35
Vlayer-85
Wlayer-86
Xlayer-87
Ylayer_with_weights-36
Ylayer-88
Zlayer-89
[layer_with_weights-37
[layer-90
\layer-91
]layer_with_weights-38
]layer-92
^regularization_losses
_trainable_variables
`	variables
a	keras_api
b
signatures
Ú__call__
+Û&call_and_return_all_conditional_losses
Ü_default_save_signature"´Ê
_tf_keras_networkÊ{"class_name": "Functional", "name": "ssi_res_unet", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "ssi_res_unet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 29, 256, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "input_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "input_conv", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d", "inbound_nodes": [[["input_conv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_1", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv1", "inbound_nodes": [[["downsampler_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_1_relu1", "inbound_nodes": [[["resblock_part1_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv2", "inbound_nodes": [[["resblock_part1_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["tf.math.multiply", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv1", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_2_relu1", "inbound_nodes": [[["resblock_part1_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv2", "inbound_nodes": [[["resblock_part1_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"y": ["tf.__operators__.add", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_3_relu1", "inbound_nodes": [[["resblock_part1_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv2", "inbound_nodes": [[["resblock_part1_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"y": ["tf.__operators__.add_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv1", "inbound_nodes": [[["tf.__operators__.add_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_4_relu1", "inbound_nodes": [[["resblock_part1_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv2", "inbound_nodes": [[["resblock_part1_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"y": ["tf.__operators__.add_2", 0, 0], "name": null}]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d_1", "inbound_nodes": [[["tf.__operators__.add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_2", "inbound_nodes": [[["zero_padding2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv1", "inbound_nodes": [[["downsampler_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_1_relu1", "inbound_nodes": [[["resblock_part2_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv2", "inbound_nodes": [[["resblock_part2_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"y": ["downsampler_2", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv1", "inbound_nodes": [[["tf.__operators__.add_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_2_relu1", "inbound_nodes": [[["resblock_part2_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv2", "inbound_nodes": [[["resblock_part2_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"y": ["tf.__operators__.add_4", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv1", "inbound_nodes": [[["tf.__operators__.add_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_3_relu1", "inbound_nodes": [[["resblock_part2_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv2", "inbound_nodes": [[["resblock_part2_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"y": ["tf.__operators__.add_5", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv1", "inbound_nodes": [[["tf.__operators__.add_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_4_relu1", "inbound_nodes": [[["resblock_part2_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv2", "inbound_nodes": [[["resblock_part2_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"y": ["tf.__operators__.add_6", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv1", "inbound_nodes": [[["tf.__operators__.add_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_5_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_5_relu1", "inbound_nodes": [[["resblock_part2_5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv2", "inbound_nodes": [[["resblock_part2_5_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_5_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"y": ["tf.__operators__.add_7", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv1", "inbound_nodes": [[["tf.__operators__.add_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_6_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_6_relu1", "inbound_nodes": [[["resblock_part2_6_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv2", "inbound_nodes": [[["resblock_part2_6_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_6_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_9", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"y": ["tf.__operators__.add_8", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv1", "inbound_nodes": [[["tf.__operators__.add_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_7_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_7_relu1", "inbound_nodes": [[["resblock_part2_7_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv2", "inbound_nodes": [[["resblock_part2_7_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_10", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_7_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_10", "inbound_nodes": [["tf.math.multiply_10", 0, 0, {"y": ["tf.__operators__.add_9", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv1", "inbound_nodes": [[["tf.__operators__.add_10", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_8_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_8_relu1", "inbound_nodes": [[["resblock_part2_8_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv2", "inbound_nodes": [[["resblock_part2_8_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_11", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_8_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_11", "inbound_nodes": [["tf.math.multiply_11", 0, 0, {"y": ["tf.__operators__.add_10", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_1", "inbound_nodes": [[["tf.__operators__.add_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space", "inbound_nodes": [["upsampler_1", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv1", "inbound_nodes": [[["tf.nn.depth_to_space", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_1_relu1", "inbound_nodes": [[["resblock_part3_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv2", "inbound_nodes": [[["resblock_part3_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_12", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_12", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["tf.math.multiply_12", 0, 0, {"y": ["tf.nn.depth_to_space", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv1", "inbound_nodes": [[["tf.__operators__.add_12", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_2_relu1", "inbound_nodes": [[["resblock_part3_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv2", "inbound_nodes": [[["resblock_part3_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_13", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_13", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.math.multiply_13", 0, 0, {"y": ["tf.__operators__.add_12", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv1", "inbound_nodes": [[["tf.__operators__.add_13", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_3_relu1", "inbound_nodes": [[["resblock_part3_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv2", "inbound_nodes": [[["resblock_part3_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_14", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_14", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["tf.math.multiply_14", 0, 0, {"y": ["tf.__operators__.add_13", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv1", "inbound_nodes": [[["tf.__operators__.add_14", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_4_relu1", "inbound_nodes": [[["resblock_part3_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv2", "inbound_nodes": [[["resblock_part3_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_15", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.math.multiply_15", 0, 0, {"y": ["tf.__operators__.add_14", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "extra_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "extra_conv", "inbound_nodes": [[["tf.__operators__.add_15", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["extra_conv", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_2", "inbound_nodes": [[["tf.__operators__.add_16", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space_1", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space_1", "inbound_nodes": [["upsampler_2", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "output_conv", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_conv", "inbound_nodes": [[["tf.nn.depth_to_space_1", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_conv", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 29, 256, 256]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 29, 256, 256]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "ssi_res_unet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 29, 256, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "input_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "input_conv", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d", "inbound_nodes": [[["input_conv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_1", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv1", "inbound_nodes": [[["downsampler_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_1_relu1", "inbound_nodes": [[["resblock_part1_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv2", "inbound_nodes": [[["resblock_part1_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["tf.math.multiply", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv1", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_2_relu1", "inbound_nodes": [[["resblock_part1_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv2", "inbound_nodes": [[["resblock_part1_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"y": ["tf.__operators__.add", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_3_relu1", "inbound_nodes": [[["resblock_part1_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv2", "inbound_nodes": [[["resblock_part1_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"y": ["tf.__operators__.add_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv1", "inbound_nodes": [[["tf.__operators__.add_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_4_relu1", "inbound_nodes": [[["resblock_part1_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv2", "inbound_nodes": [[["resblock_part1_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"y": ["tf.__operators__.add_2", 0, 0], "name": null}]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d_1", "inbound_nodes": [[["tf.__operators__.add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_2", "inbound_nodes": [[["zero_padding2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv1", "inbound_nodes": [[["downsampler_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_1_relu1", "inbound_nodes": [[["resblock_part2_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv2", "inbound_nodes": [[["resblock_part2_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"y": ["downsampler_2", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv1", "inbound_nodes": [[["tf.__operators__.add_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_2_relu1", "inbound_nodes": [[["resblock_part2_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv2", "inbound_nodes": [[["resblock_part2_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"y": ["tf.__operators__.add_4", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv1", "inbound_nodes": [[["tf.__operators__.add_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_3_relu1", "inbound_nodes": [[["resblock_part2_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv2", "inbound_nodes": [[["resblock_part2_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"y": ["tf.__operators__.add_5", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv1", "inbound_nodes": [[["tf.__operators__.add_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_4_relu1", "inbound_nodes": [[["resblock_part2_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv2", "inbound_nodes": [[["resblock_part2_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"y": ["tf.__operators__.add_6", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv1", "inbound_nodes": [[["tf.__operators__.add_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_5_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_5_relu1", "inbound_nodes": [[["resblock_part2_5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv2", "inbound_nodes": [[["resblock_part2_5_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_5_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"y": ["tf.__operators__.add_7", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv1", "inbound_nodes": [[["tf.__operators__.add_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_6_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_6_relu1", "inbound_nodes": [[["resblock_part2_6_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv2", "inbound_nodes": [[["resblock_part2_6_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_6_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_9", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"y": ["tf.__operators__.add_8", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv1", "inbound_nodes": [[["tf.__operators__.add_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_7_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_7_relu1", "inbound_nodes": [[["resblock_part2_7_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv2", "inbound_nodes": [[["resblock_part2_7_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_10", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_7_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_10", "inbound_nodes": [["tf.math.multiply_10", 0, 0, {"y": ["tf.__operators__.add_9", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv1", "inbound_nodes": [[["tf.__operators__.add_10", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_8_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_8_relu1", "inbound_nodes": [[["resblock_part2_8_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv2", "inbound_nodes": [[["resblock_part2_8_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_11", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_8_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_11", "inbound_nodes": [["tf.math.multiply_11", 0, 0, {"y": ["tf.__operators__.add_10", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_1", "inbound_nodes": [[["tf.__operators__.add_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space", "inbound_nodes": [["upsampler_1", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv1", "inbound_nodes": [[["tf.nn.depth_to_space", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_1_relu1", "inbound_nodes": [[["resblock_part3_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv2", "inbound_nodes": [[["resblock_part3_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_12", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_12", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["tf.math.multiply_12", 0, 0, {"y": ["tf.nn.depth_to_space", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv1", "inbound_nodes": [[["tf.__operators__.add_12", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_2_relu1", "inbound_nodes": [[["resblock_part3_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv2", "inbound_nodes": [[["resblock_part3_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_13", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_13", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.math.multiply_13", 0, 0, {"y": ["tf.__operators__.add_12", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv1", "inbound_nodes": [[["tf.__operators__.add_13", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_3_relu1", "inbound_nodes": [[["resblock_part3_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv2", "inbound_nodes": [[["resblock_part3_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_14", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_14", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["tf.math.multiply_14", 0, 0, {"y": ["tf.__operators__.add_13", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv1", "inbound_nodes": [[["tf.__operators__.add_14", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_4_relu1", "inbound_nodes": [[["resblock_part3_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv2", "inbound_nodes": [[["resblock_part3_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_15", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.math.multiply_15", 0, 0, {"y": ["tf.__operators__.add_14", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "extra_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "extra_conv", "inbound_nodes": [[["tf.__operators__.add_15", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["extra_conv", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_2", "inbound_nodes": [[["tf.__operators__.add_16", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space_1", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space_1", "inbound_nodes": [["upsampler_2", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "output_conv", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_conv", "inbound_nodes": [[["tf.nn.depth_to_space_1", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_conv", 0, 0]]}}}
"
_tf_keras_input_layerä{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 29, 256, 256]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 29, 256, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
ý	

ckernel
dbias
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Conv2D", "name": "input_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "input_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 29}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 29, 256, 256]}}

iregularization_losses
jtrainable_variables
k	variables
l	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"÷
_tf_keras_layerÝ{"class_name": "ZeroPadding2D", "name": "zero_padding2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}



mkernel
nbias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
á__call__
+â&call_and_return_all_conditional_losses"Ý
_tf_keras_layerÃ{"class_name": "Conv2D", "name": "downsampler_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "downsampler_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 258, 258]}}



skernel
tbias
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

yregularization_losses
ztrainable_variables
{	variables
|	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_1_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}



}kernel
~bias
regularization_losses
trainable_variables
	variables
	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_1_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
ç
	keras_api"Ô
_tf_keras_layerº{"class_name": "TFOpLambda", "name": "tf.math.multiply", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ó
	keras_api"à
_tf_keras_layerÆ{"class_name": "TFOpLambda", "name": "tf.__operators__.add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
é__call__
+ê&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

regularization_losses
trainable_variables
	variables
	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_2_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
í__call__
+î&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_2_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
ë
	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
ï__call__
+ð&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

regularization_losses
trainable_variables
	variables
 	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_3_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


¡kernel
	¢bias
£regularization_losses
¤trainable_variables
¥	variables
¦	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_3_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
ë
§	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
¨	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


©kernel
	ªbias
«regularization_losses
¬trainable_variables
­	variables
®	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

¯regularization_losses
°trainable_variables
±	variables
²	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_4_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


³kernel
	´bias
µregularization_losses
¶trainable_variables
·	variables
¸	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_4_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
ë
¹	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
º	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}

»regularization_losses
¼trainable_variables
½	variables
¾	keras_api
û__call__
+ü&call_and_return_all_conditional_losses"û
_tf_keras_layerá{"class_name": "ZeroPadding2D", "name": "zero_padding2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}


¿kernel
	Àbias
Áregularization_losses
Âtrainable_variables
Ã	variables
Ä	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"Ý
_tf_keras_layerÃ{"class_name": "Conv2D", "name": "downsampler_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "downsampler_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 130, 130]}}


Åkernel
	Æbias
Çregularization_losses
Ètrainable_variables
É	variables
Ê	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

Ëregularization_losses
Ìtrainable_variables
Í	variables
Î	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_1_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


Ïkernel
	Ðbias
Ñregularization_losses
Òtrainable_variables
Ó	variables
Ô	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_1_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
Õ	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
Ö	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


×kernel
	Øbias
Ùregularization_losses
Útrainable_variables
Û	variables
Ü	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

Ýregularization_losses
Þtrainable_variables
ß	variables
à	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_2_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


ákernel
	âbias
ãregularization_losses
ätrainable_variables
å	variables
æ	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_2_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
ç	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
è	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


ékernel
	êbias
ëregularization_losses
ìtrainable_variables
í	variables
î	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

ïregularization_losses
ðtrainable_variables
ñ	variables
ò	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_3_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


ókernel
	ôbias
õregularization_losses
ötrainable_variables
÷	variables
ø	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_3_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
ù	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
ú	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


ûkernel
	übias
ýregularization_losses
þtrainable_variables
ÿ	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_4_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_4_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_5_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_5_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_5_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_5_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_5_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_5_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	 bias
¡regularization_losses
¢trainable_variables
£	variables
¤	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_6_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_6_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

¥regularization_losses
¦trainable_variables
§	variables
¨	keras_api
__call__
+ &call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_6_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_6_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


©kernel
	ªbias
«regularization_losses
¬trainable_variables
­	variables
®	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_6_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_6_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ë
¯	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
÷
°	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


±kernel
	²bias
³regularization_losses
´trainable_variables
µ	variables
¶	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_7_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_7_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

·regularization_losses
¸trainable_variables
¹	variables
º	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_7_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_7_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


»kernel
	¼bias
½regularization_losses
¾trainable_variables
¿	variables
À	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_7_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_7_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
í
Á	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
Â	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


Ãkernel
	Äbias
Åregularization_losses
Ætrainable_variables
Ç	variables
È	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_8_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_8_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

Éregularization_losses
Êtrainable_variables
Ë	variables
Ì	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_8_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_8_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


Íkernel
	Îbias
Ïregularization_losses
Ðtrainable_variables
Ñ	variables
Ò	keras_api
­__call__
+®&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_8_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_8_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
í
Ó	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
Ô	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


Õkernel
	Öbias
×regularization_losses
Øtrainable_variables
Ù	variables
Ú	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Conv2D", "name": "upsampler_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsampler_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
ó
Û	keras_api"à
_tf_keras_layerÆ{"class_name": "TFOpLambda", "name": "tf.nn.depth_to_space", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.depth_to_space", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}}


Ükernel
	Ýbias
Þregularization_losses
ßtrainable_variables
à	variables
á	keras_api
±__call__
+²&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

âregularization_losses
ãtrainable_variables
ä	variables
å	keras_api
³__call__
+´&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_1_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


ækernel
	çbias
èregularization_losses
étrainable_variables
ê	variables
ë	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_1_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
í
ì	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_12", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
í	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


îkernel
	ïbias
ðregularization_losses
ñtrainable_variables
ò	variables
ó	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

ôregularization_losses
õtrainable_variables
ö	variables
÷	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_2_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


økernel
	ùbias
úregularization_losses
ûtrainable_variables
ü	variables
ý	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_2_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
í
þ	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_13", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
ÿ	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
½__call__
+¾&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

regularization_losses
trainable_variables
	variables
	keras_api
¿__call__
+À&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_3_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_3_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
í
	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_14", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


kernel
	bias
regularization_losses
trainable_variables
	variables
	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

regularization_losses
trainable_variables
	variables
	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_4_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
regularization_losses
trainable_variables
 	variables
¡	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_4_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
í
¢	keras_api"Ú
_tf_keras_layerÀ{"class_name": "TFOpLambda", "name": "tf.math.multiply_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ù
£	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


¤kernel
	¥bias
¦regularization_losses
§trainable_variables
¨	variables
©	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Conv2D", "name": "extra_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "extra_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
ù
ª	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}


«kernel
	¬bias
­regularization_losses
®trainable_variables
¯	variables
°	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"Ù
_tf_keras_layer¿{"class_name": "Conv2D", "name": "upsampler_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsampler_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
÷
±	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.nn.depth_to_space_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.depth_to_space_1", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}}


²kernel
	³bias
´regularization_losses
µtrainable_variables
¶	variables
·	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Conv2D", "name": "output_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_conv", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 256, 256]}}
 "
trackable_list_wrapper
Ì
c0
d1
m2
n3
s4
t5
}6
~7
8
9
10
11
12
13
¡14
¢15
©16
ª17
³18
´19
¿20
À21
Å22
Æ23
Ï24
Ð25
×26
Ø27
á28
â29
é30
ê31
ó32
ô33
û34
ü35
36
37
38
39
40
41
42
 43
©44
ª45
±46
²47
»48
¼49
Ã50
Ä51
Í52
Î53
Õ54
Ö55
Ü56
Ý57
æ58
ç59
î60
ï61
ø62
ù63
64
65
66
67
68
69
70
71
¤72
¥73
«74
¬75
²76
³77"
trackable_list_wrapper
Ì
c0
d1
m2
n3
s4
t5
}6
~7
8
9
10
11
12
13
¡14
¢15
©16
ª17
³18
´19
¿20
À21
Å22
Æ23
Ï24
Ð25
×26
Ø27
á28
â29
é30
ê31
ó32
ô33
û34
ü35
36
37
38
39
40
41
42
 43
©44
ª45
±46
²47
»48
¼49
Ã50
Ä51
Í52
Î53
Õ54
Ö55
Ü56
Ý57
æ58
ç59
î60
ï61
ø62
ù63
64
65
66
67
68
69
70
71
¤72
¥73
«74
¬75
²76
³77"
trackable_list_wrapper
Ó
^regularization_losses
¸layers
_trainable_variables
`	variables
¹metrics
ºnon_trainable_variables
»layer_metrics
 ¼layer_regularization_losses
Ú__call__
Ü_default_save_signature
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
-
Ïserving_default"
signature_map
+:)@2input_conv/kernel
:@2input_conv/bias
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
µ
eregularization_losses
½layers
ftrainable_variables
g	variables
¾metrics
¿non_trainable_variables
Àlayer_metrics
 Álayer_regularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
iregularization_losses
Âlayers
jtrainable_variables
k	variables
Ãmetrics
Änon_trainable_variables
Ålayer_metrics
 Ælayer_regularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
.:,@@2downsampler_1/kernel
 :@2downsampler_1/bias
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
µ
oregularization_losses
Çlayers
ptrainable_variables
q	variables
Èmetrics
Énon_trainable_variables
Êlayer_metrics
 Ëlayer_regularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_1_conv1/kernel
):'@2resblock_part1_1_conv1/bias
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
µ
uregularization_losses
Ìlayers
vtrainable_variables
w	variables
Ímetrics
Înon_trainable_variables
Ïlayer_metrics
 Ðlayer_regularization_losses
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
yregularization_losses
Ñlayers
ztrainable_variables
{	variables
Òmetrics
Ónon_trainable_variables
Ôlayer_metrics
 Õlayer_regularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_1_conv2/kernel
):'@2resblock_part1_1_conv2/bias
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
·
regularization_losses
Ölayers
trainable_variables
	variables
×metrics
Ønon_trainable_variables
Ùlayer_metrics
 Úlayer_regularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part1_2_conv1/kernel
):'@2resblock_part1_2_conv1/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
Ûlayers
trainable_variables
	variables
Ümetrics
Ýnon_trainable_variables
Þlayer_metrics
 ßlayer_regularization_losses
é__call__
+ê&call_and_return_all_conditional_losses
'ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
àlayers
trainable_variables
	variables
ámetrics
ânon_trainable_variables
ãlayer_metrics
 älayer_regularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_2_conv2/kernel
):'@2resblock_part1_2_conv2/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
ålayers
trainable_variables
	variables
æmetrics
çnon_trainable_variables
èlayer_metrics
 élayer_regularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part1_3_conv1/kernel
):'@2resblock_part1_3_conv1/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
êlayers
trainable_variables
	variables
ëmetrics
ìnon_trainable_variables
ílayer_metrics
 îlayer_regularization_losses
ï__call__
+ð&call_and_return_all_conditional_losses
'ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
ïlayers
trainable_variables
	variables
ðmetrics
ñnon_trainable_variables
òlayer_metrics
 ólayer_regularization_losses
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_3_conv2/kernel
):'@2resblock_part1_3_conv2/bias
 "
trackable_list_wrapper
0
¡0
¢1"
trackable_list_wrapper
0
¡0
¢1"
trackable_list_wrapper
¸
£regularization_losses
ôlayers
¤trainable_variables
¥	variables
õmetrics
önon_trainable_variables
÷layer_metrics
 ølayer_regularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part1_4_conv1/kernel
):'@2resblock_part1_4_conv1/bias
 "
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
¸
«regularization_losses
ùlayers
¬trainable_variables
­	variables
úmetrics
ûnon_trainable_variables
ülayer_metrics
 ýlayer_regularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¯regularization_losses
þlayers
°trainable_variables
±	variables
ÿmetrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_4_conv2/kernel
):'@2resblock_part1_4_conv2/bias
 "
trackable_list_wrapper
0
³0
´1"
trackable_list_wrapper
0
³0
´1"
trackable_list_wrapper
¸
µregularization_losses
layers
¶trainable_variables
·	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
»regularization_losses
layers
¼trainable_variables
½	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
.:,@@2downsampler_2/kernel
 :@2downsampler_2/bias
 "
trackable_list_wrapper
0
¿0
À1"
trackable_list_wrapper
0
¿0
À1"
trackable_list_wrapper
¸
Áregularization_losses
layers
Âtrainable_variables
Ã	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_1_conv1/kernel
):'@2resblock_part2_1_conv1/bias
 "
trackable_list_wrapper
0
Å0
Æ1"
trackable_list_wrapper
0
Å0
Æ1"
trackable_list_wrapper
¸
Çregularization_losses
layers
Ètrainable_variables
É	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ëregularization_losses
layers
Ìtrainable_variables
Í	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_1_conv2/kernel
):'@2resblock_part2_1_conv2/bias
 "
trackable_list_wrapper
0
Ï0
Ð1"
trackable_list_wrapper
0
Ï0
Ð1"
trackable_list_wrapper
¸
Ñregularization_losses
layers
Òtrainable_variables
Ó	variables
metrics
non_trainable_variables
layer_metrics
  layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_2_conv1/kernel
):'@2resblock_part2_2_conv1/bias
 "
trackable_list_wrapper
0
×0
Ø1"
trackable_list_wrapper
0
×0
Ø1"
trackable_list_wrapper
¸
Ùregularization_losses
¡layers
Útrainable_variables
Û	variables
¢metrics
£non_trainable_variables
¤layer_metrics
 ¥layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ýregularization_losses
¦layers
Þtrainable_variables
ß	variables
§metrics
¨non_trainable_variables
©layer_metrics
 ªlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_2_conv2/kernel
):'@2resblock_part2_2_conv2/bias
 "
trackable_list_wrapper
0
á0
â1"
trackable_list_wrapper
0
á0
â1"
trackable_list_wrapper
¸
ãregularization_losses
«layers
ätrainable_variables
å	variables
¬metrics
­non_trainable_variables
®layer_metrics
 ¯layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_3_conv1/kernel
):'@2resblock_part2_3_conv1/bias
 "
trackable_list_wrapper
0
é0
ê1"
trackable_list_wrapper
0
é0
ê1"
trackable_list_wrapper
¸
ëregularization_losses
°layers
ìtrainable_variables
í	variables
±metrics
²non_trainable_variables
³layer_metrics
 ´layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïregularization_losses
µlayers
ðtrainable_variables
ñ	variables
¶metrics
·non_trainable_variables
¸layer_metrics
 ¹layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_3_conv2/kernel
):'@2resblock_part2_3_conv2/bias
 "
trackable_list_wrapper
0
ó0
ô1"
trackable_list_wrapper
0
ó0
ô1"
trackable_list_wrapper
¸
õregularization_losses
ºlayers
ötrainable_variables
÷	variables
»metrics
¼non_trainable_variables
½layer_metrics
 ¾layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_4_conv1/kernel
):'@2resblock_part2_4_conv1/bias
 "
trackable_list_wrapper
0
û0
ü1"
trackable_list_wrapper
0
û0
ü1"
trackable_list_wrapper
¸
ýregularization_losses
¿layers
þtrainable_variables
ÿ	variables
Àmetrics
Ánon_trainable_variables
Âlayer_metrics
 Ãlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
Älayers
trainable_variables
	variables
Åmetrics
Ænon_trainable_variables
Çlayer_metrics
 Èlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_4_conv2/kernel
):'@2resblock_part2_4_conv2/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
Élayers
trainable_variables
	variables
Êmetrics
Ënon_trainable_variables
Ìlayer_metrics
 Ílayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_5_conv1/kernel
):'@2resblock_part2_5_conv1/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
Îlayers
trainable_variables
	variables
Ïmetrics
Ðnon_trainable_variables
Ñlayer_metrics
 Òlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
Ólayers
trainable_variables
	variables
Ômetrics
Õnon_trainable_variables
Ölayer_metrics
 ×layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_5_conv2/kernel
):'@2resblock_part2_5_conv2/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
Ølayers
trainable_variables
	variables
Ùmetrics
Únon_trainable_variables
Ûlayer_metrics
 Ülayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_6_conv1/kernel
):'@2resblock_part2_6_conv1/bias
 "
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
¸
¡regularization_losses
Ýlayers
¢trainable_variables
£	variables
Þmetrics
ßnon_trainable_variables
àlayer_metrics
 álayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¥regularization_losses
âlayers
¦trainable_variables
§	variables
ãmetrics
änon_trainable_variables
ålayer_metrics
 ælayer_regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_6_conv2/kernel
):'@2resblock_part2_6_conv2/bias
 "
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
¸
«regularization_losses
çlayers
¬trainable_variables
­	variables
èmetrics
énon_trainable_variables
êlayer_metrics
 ëlayer_regularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_7_conv1/kernel
):'@2resblock_part2_7_conv1/bias
 "
trackable_list_wrapper
0
±0
²1"
trackable_list_wrapper
0
±0
²1"
trackable_list_wrapper
¸
³regularization_losses
ìlayers
´trainable_variables
µ	variables
ímetrics
înon_trainable_variables
ïlayer_metrics
 ðlayer_regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
·regularization_losses
ñlayers
¸trainable_variables
¹	variables
òmetrics
ónon_trainable_variables
ôlayer_metrics
 õlayer_regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_7_conv2/kernel
):'@2resblock_part2_7_conv2/bias
 "
trackable_list_wrapper
0
»0
¼1"
trackable_list_wrapper
0
»0
¼1"
trackable_list_wrapper
¸
½regularization_losses
ölayers
¾trainable_variables
¿	variables
÷metrics
ønon_trainable_variables
ùlayer_metrics
 úlayer_regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part2_8_conv1/kernel
):'@2resblock_part2_8_conv1/bias
 "
trackable_list_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
¸
Åregularization_losses
ûlayers
Ætrainable_variables
Ç	variables
ümetrics
ýnon_trainable_variables
þlayer_metrics
 ÿlayer_regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Éregularization_losses
layers
Êtrainable_variables
Ë	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_8_conv2/kernel
):'@2resblock_part2_8_conv2/bias
 "
trackable_list_wrapper
0
Í0
Î1"
trackable_list_wrapper
0
Í0
Î1"
trackable_list_wrapper
¸
Ïregularization_losses
layers
Ðtrainable_variables
Ñ	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
-:+@2upsampler_1/kernel
:2upsampler_1/bias
 "
trackable_list_wrapper
0
Õ0
Ö1"
trackable_list_wrapper
0
Õ0
Ö1"
trackable_list_wrapper
¸
×regularization_losses
layers
Øtrainable_variables
Ù	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part3_1_conv1/kernel
):'@2resblock_part3_1_conv1/bias
 "
trackable_list_wrapper
0
Ü0
Ý1"
trackable_list_wrapper
0
Ü0
Ý1"
trackable_list_wrapper
¸
Þregularization_losses
layers
ßtrainable_variables
à	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
âregularization_losses
layers
ãtrainable_variables
ä	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_1_conv2/kernel
):'@2resblock_part3_1_conv2/bias
 "
trackable_list_wrapper
0
æ0
ç1"
trackable_list_wrapper
0
æ0
ç1"
trackable_list_wrapper
¸
èregularization_losses
layers
étrainable_variables
ê	variables
metrics
non_trainable_variables
layer_metrics
 layer_regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part3_2_conv1/kernel
):'@2resblock_part3_2_conv1/bias
 "
trackable_list_wrapper
0
î0
ï1"
trackable_list_wrapper
0
î0
ï1"
trackable_list_wrapper
¸
ðregularization_losses
layers
ñtrainable_variables
ò	variables
metrics
 non_trainable_variables
¡layer_metrics
 ¢layer_regularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ôregularization_losses
£layers
õtrainable_variables
ö	variables
¤metrics
¥non_trainable_variables
¦layer_metrics
 §layer_regularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_2_conv2/kernel
):'@2resblock_part3_2_conv2/bias
 "
trackable_list_wrapper
0
ø0
ù1"
trackable_list_wrapper
0
ø0
ù1"
trackable_list_wrapper
¸
úregularization_losses
¨layers
ûtrainable_variables
ü	variables
©metrics
ªnon_trainable_variables
«layer_metrics
 ¬layer_regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part3_3_conv1/kernel
):'@2resblock_part3_3_conv1/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
­layers
trainable_variables
	variables
®metrics
¯non_trainable_variables
°layer_metrics
 ±layer_regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
²layers
trainable_variables
	variables
³metrics
´non_trainable_variables
µlayer_metrics
 ¶layer_regularization_losses
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_3_conv2/kernel
):'@2resblock_part3_3_conv2/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
·layers
trainable_variables
	variables
¸metrics
¹non_trainable_variables
ºlayer_metrics
 »layer_regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part3_4_conv1/kernel
):'@2resblock_part3_4_conv1/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
¼layers
trainable_variables
	variables
½metrics
¾non_trainable_variables
¿layer_metrics
 Àlayer_regularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
Álayers
trainable_variables
	variables
Âmetrics
Ãnon_trainable_variables
Älayer_metrics
 Ålayer_regularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_4_conv2/kernel
):'@2resblock_part3_4_conv2/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
regularization_losses
Ælayers
trainable_variables
 	variables
Çmetrics
Ènon_trainable_variables
Élayer_metrics
 Êlayer_regularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
+:)@@2extra_conv/kernel
:@2extra_conv/bias
 "
trackable_list_wrapper
0
¤0
¥1"
trackable_list_wrapper
0
¤0
¥1"
trackable_list_wrapper
¸
¦regularization_losses
Ëlayers
§trainable_variables
¨	variables
Ìmetrics
Ínon_trainable_variables
Îlayer_metrics
 Ïlayer_regularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
-:+@2upsampler_2/kernel
:2upsampler_2/bias
 "
trackable_list_wrapper
0
«0
¬1"
trackable_list_wrapper
0
«0
¬1"
trackable_list_wrapper
¸
­regularization_losses
Ðlayers
®trainable_variables
¯	variables
Ñmetrics
Ònon_trainable_variables
Ólayer_metrics
 Ôlayer_regularization_losses
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
,:*@2output_conv/kernel
:2output_conv/bias
 "
trackable_list_wrapper
0
²0
³1"
trackable_list_wrapper
0
²0
³1"
trackable_list_wrapper
¸
´regularization_losses
Õlayers
µtrainable_variables
¶	variables
Ömetrics
×non_trainable_variables
Ølayer_metrics
 Ùlayer_regularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
þ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
W86
X87
Y88
Z89
[90
\91
]92"
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
ú2÷
+__inference_ssi_res_unet_layer_call_fn_5754
+__inference_ssi_res_unet_layer_call_fn_4094
+__inference_ssi_res_unet_layer_call_fn_5561
+__inference_ssi_res_unet_layer_call_fn_4555À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_3632
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5059
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5368
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_3364À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
__inference__wrapped_model_2058Ä
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *4¢1
/,
input_layerÿÿÿÿÿÿÿÿÿ
Ó2Ð
)__inference_input_conv_layer_call_fn_5773¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_input_conv_layer_call_and_return_conditional_losses_5764¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
-__inference_zero_padding2d_layer_call_fn_2071à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°2­
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_2065à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ö2Ó
,__inference_downsampler_1_layer_call_fn_5792¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_downsampler_1_layer_call_and_return_conditional_losses_5783¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_1_conv1_layer_call_fn_5811¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_5802¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_1_relu1_layer_call_fn_5821¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_5816¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_1_conv2_layer_call_fn_5840¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_5831¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_2_conv1_layer_call_fn_5859¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_5850¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_2_relu1_layer_call_fn_5869¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_5864¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_2_conv2_layer_call_fn_5888¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_5879¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_3_conv1_layer_call_fn_5907¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_5898¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_3_relu1_layer_call_fn_5917¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_5912¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_3_conv2_layer_call_fn_5936¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_5927¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_4_conv1_layer_call_fn_5955¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_5946¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_4_relu1_layer_call_fn_5965¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_5960¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part1_4_conv2_layer_call_fn_5984¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_5975¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
/__inference_zero_padding2d_1_layer_call_fn_2084à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
²2¯
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2078à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ö2Ó
,__inference_downsampler_2_layer_call_fn_6003¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_downsampler_2_layer_call_and_return_conditional_losses_5994¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_1_conv1_layer_call_fn_6022¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_6013¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_1_relu1_layer_call_fn_6032¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_6027¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_1_conv2_layer_call_fn_6051¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_6042¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_2_conv1_layer_call_fn_6070¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_6061¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_2_relu1_layer_call_fn_6080¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_6075¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_2_conv2_layer_call_fn_6099¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_6090¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_3_conv1_layer_call_fn_6118¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_6109¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_3_relu1_layer_call_fn_6128¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_6123¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_3_conv2_layer_call_fn_6147¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_6138¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_4_conv1_layer_call_fn_6166¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_6157¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_4_relu1_layer_call_fn_6176¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_6171¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_4_conv2_layer_call_fn_6195¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_6186¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_5_conv1_layer_call_fn_6214¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_6205¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_5_relu1_layer_call_fn_6224¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_6219¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_5_conv2_layer_call_fn_6243¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_6234¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_6_conv1_layer_call_fn_6262¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_6253¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_6_relu1_layer_call_fn_6272¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_6267¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_6_conv2_layer_call_fn_6291¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_6282¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_7_conv1_layer_call_fn_6310¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_6301¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_7_relu1_layer_call_fn_6320¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_6315¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_7_conv2_layer_call_fn_6339¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_6330¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_8_conv1_layer_call_fn_6358¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_6349¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_8_relu1_layer_call_fn_6368¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_6363¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part2_8_conv2_layer_call_fn_6387¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_6378¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_upsampler_1_layer_call_fn_6406¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_upsampler_1_layer_call_and_return_conditional_losses_6397¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_1_conv1_layer_call_fn_6425¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_6416¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_1_relu1_layer_call_fn_6435¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_6430¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_1_conv2_layer_call_fn_6454¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_6445¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_2_conv1_layer_call_fn_6473¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_6464¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_2_relu1_layer_call_fn_6483¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_6478¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_2_conv2_layer_call_fn_6502¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_6493¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_3_conv1_layer_call_fn_6521¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_6512¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_3_relu1_layer_call_fn_6531¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_6526¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_3_conv2_layer_call_fn_6550¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_6541¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_4_conv1_layer_call_fn_6569¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_6560¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_4_relu1_layer_call_fn_6579¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_6574¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_resblock_part3_4_conv2_layer_call_fn_6598¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_6589¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_extra_conv_layer_call_fn_6617¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_extra_conv_layer_call_and_return_conditional_losses_6608¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_upsampler_2_layer_call_fn_6636¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_upsampler_2_layer_call_and_return_conditional_losses_6627¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_output_conv_layer_call_fn_6655¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_output_conv_layer_call_and_return_conditional_losses_6646¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
"__inference_signature_wrapper_4750input_layer"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9
J

Const_10
J

Const_11
J

Const_12
J

Const_13
J

Const_14
J

Const_15à
__inference__wrapped_model_2058¼´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³>¢;
4¢1
/,
input_layerÿÿÿÿÿÿÿÿÿ
ª "Cª@
>
output_conv/,
output_convÿÿÿÿÿÿÿÿÿ»
G__inference_downsampler_1_layer_call_and_return_conditional_losses_5783pmn9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_downsampler_1_layer_call_fn_5792cmn9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@»
G__inference_downsampler_2_layer_call_and_return_conditional_losses_5994p¿À9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
,__inference_downsampler_2_layer_call_fn_6003c¿À9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@@@º
D__inference_extra_conv_layer_call_and_return_conditional_losses_6608r¤¥9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_extra_conv_layer_call_fn_6617e¤¥9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@¸
D__inference_input_conv_layer_call_and_return_conditional_losses_5764pcd9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_input_conv_layer_call_fn_5773ccd9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ@»
E__inference_output_conv_layer_call_and_return_conditional_losses_6646r²³9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_output_conv_layer_call_fn_6655e²³9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿÄ
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_5802pst9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_1_conv1_layer_call_fn_5811cst9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Ä
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_5831p}~9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_1_conv2_layer_call_fn_5840c}~9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_5816l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_1_relu1_layer_call_fn_5821_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_5850r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_2_conv1_layer_call_fn_5859e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_5879r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_2_conv2_layer_call_fn_5888e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_5864l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_2_relu1_layer_call_fn_5869_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_5898r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_3_conv1_layer_call_fn_5907e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_5927r¡¢9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_3_conv2_layer_call_fn_5936e¡¢9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_5912l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_3_relu1_layer_call_fn_5917_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_5946r©ª9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_4_conv1_layer_call_fn_5955e©ª9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_5975r³´9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_4_conv2_layer_call_fn_5984e³´9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_5960l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_4_relu1_layer_call_fn_5965_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Â
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_6013nÅÆ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_1_conv1_layer_call_fn_6022aÅÆ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_6042nÏÐ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_1_conv2_layer_call_fn_6051aÏÐ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_6027h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_1_relu1_layer_call_fn_6032[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_6061n×Ø7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_2_conv1_layer_call_fn_6070a×Ø7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_6090náâ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_2_conv2_layer_call_fn_6099aáâ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_6075h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_2_relu1_layer_call_fn_6080[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_6109néê7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_3_conv1_layer_call_fn_6118aéê7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_6138nóô7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_3_conv2_layer_call_fn_6147aóô7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_6123h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_3_relu1_layer_call_fn_6128[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_6157nûü7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_4_conv1_layer_call_fn_6166aûü7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_6186n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_4_conv2_layer_call_fn_6195a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_6171h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_4_relu1_layer_call_fn_6176[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_6205n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_5_conv1_layer_call_fn_6214a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_6234n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_5_conv2_layer_call_fn_6243a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_6219h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_5_relu1_layer_call_fn_6224[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_6253n 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_6_conv1_layer_call_fn_6262a 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_6282n©ª7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_6_conv2_layer_call_fn_6291a©ª7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_6267h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_6_relu1_layer_call_fn_6272[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_6301n±²7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_7_conv1_layer_call_fn_6310a±²7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_6330n»¼7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_7_conv2_layer_call_fn_6339a»¼7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_6315h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_7_relu1_layer_call_fn_6320[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_6349nÃÄ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_8_conv1_layer_call_fn_6358aÃÄ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_6378nÍÎ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_8_conv2_layer_call_fn_6387aÍÎ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_6363h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_8_relu1_layer_call_fn_6368[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Æ
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_6416rÜÝ9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_1_conv1_layer_call_fn_6425eÜÝ9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_6445ræç9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_1_conv2_layer_call_fn_6454eæç9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_6430l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_1_relu1_layer_call_fn_6435_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_6464rîï9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_2_conv1_layer_call_fn_6473eîï9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_6493røù9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_2_conv2_layer_call_fn_6502eøù9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_6478l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_2_relu1_layer_call_fn_6483_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_6512r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_3_conv1_layer_call_fn_6521e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_6541r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_3_conv2_layer_call_fn_6550e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_6526l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_3_relu1_layer_call_fn_6531_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_6560r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_4_conv1_layer_call_fn_6569e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_6589r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_4_conv2_layer_call_fn_6598e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_6574l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_4_relu1_layer_call_fn_6579_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@ò
"__inference_signature_wrapper_4750Ë´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³M¢J
¢ 
Cª@
>
input_layer/,
input_layerÿÿÿÿÿÿÿÿÿ"Cª@
>
output_conv/,
output_convÿÿÿÿÿÿÿÿÿû
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_3364°´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 û
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_3632°´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ö
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5059«´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ö
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5368«´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ó
+__inference_ssi_res_unet_layer_call_fn_4094£´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿÓ
+__inference_ssi_res_unet_layer_call_fn_4555£´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿÎ
+__inference_ssi_res_unet_layer_call_fn_5561´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿÎ
+__inference_ssi_res_unet_layer_call_fn_5754´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ¸
E__inference_upsampler_1_layer_call_and_return_conditional_losses_6397oÕÖ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
*__inference_upsampler_1_layer_call_fn_6406bÕÖ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "!ÿÿÿÿÿÿÿÿÿ@@¼
E__inference_upsampler_2_layer_call_and_return_conditional_losses_6627s«¬9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "0¢-
&#
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_upsampler_2_layer_call_fn_6636f«¬9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "# ÿÿÿÿÿÿÿÿÿí
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2078R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_zero_padding2d_1_layer_call_fn_2084R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_2065R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_zero_padding2d_layer_call_fn_2071R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ