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
shape:@*"
shared_nameinput_conv/kernel

%input_conv/kernel/Read/ReadVariableOpReadVariableOpinput_conv/kernel*&
_output_shapes
:@*
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
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b
signatures
 
h

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
R
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
h

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
h

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
R
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
k

}kernel
~bias
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
 	keras_api
n
¡kernel
	¢bias
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api

§	keras_api

¨	keras_api
n
©kernel
	ªbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
V
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
n
³kernel
	´bias
µ	variables
¶trainable_variables
·regularization_losses
¸	keras_api

¹	keras_api

º	keras_api
V
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
n
¿kernel
	Àbias
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
n
Åkernel
	Æbias
Ç	variables
Ètrainable_variables
Éregularization_losses
Ê	keras_api
V
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
n
Ïkernel
	Ðbias
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api

Õ	keras_api

Ö	keras_api
n
×kernel
	Øbias
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
V
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
n
ákernel
	âbias
ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api

ç	keras_api

è	keras_api
n
ékernel
	êbias
ë	variables
ìtrainable_variables
íregularization_losses
î	keras_api
V
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
n
ókernel
	ôbias
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api

ù	keras_api

ú	keras_api
n
ûkernel
	übias
ý	variables
þtrainable_variables
ÿregularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
n
kernel
	 bias
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
V
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
n
©kernel
	ªbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api

¯	keras_api

°	keras_api
n
±kernel
	²bias
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
V
·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
n
»kernel
	¼bias
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api

Á	keras_api

Â	keras_api
n
Ãkernel
	Äbias
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
V
É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
n
Íkernel
	Îbias
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api

Ó	keras_api

Ô	keras_api
n
Õkernel
	Öbias
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api

Û	keras_api
n
Ükernel
	Ýbias
Þ	variables
ßtrainable_variables
àregularization_losses
á	keras_api
V
â	variables
ãtrainable_variables
äregularization_losses
å	keras_api
n
ækernel
	çbias
è	variables
étrainable_variables
êregularization_losses
ë	keras_api

ì	keras_api

í	keras_api
n
îkernel
	ïbias
ð	variables
ñtrainable_variables
òregularization_losses
ó	keras_api
V
ô	variables
õtrainable_variables
öregularization_losses
÷	keras_api
n
økernel
	ùbias
ú	variables
ûtrainable_variables
üregularization_losses
ý	keras_api

þ	keras_api

ÿ	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
 regularization_losses
¡	keras_api

¢	keras_api

£	keras_api
n
¤kernel
	¥bias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api

ª	keras_api
n
«kernel
	¬bias
­	variables
®trainable_variables
¯regularization_losses
°	keras_api

±	keras_api
n
²kernel
	³bias
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
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
 
²
¸layers
¹layer_metrics
^	variables
ºnon_trainable_variables
»metrics
_trainable_variables
 ¼layer_regularization_losses
`regularization_losses
 
][
VARIABLE_VALUEinput_conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEinput_conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

c0
d1
 
²
½layers
¾layer_metrics
e	variables
¿non_trainable_variables
Àmetrics
ftrainable_variables
 Álayer_regularization_losses
gregularization_losses
 
 
 
²
Âlayers
Ãlayer_metrics
i	variables
Änon_trainable_variables
Åmetrics
jtrainable_variables
 Ælayer_regularization_losses
kregularization_losses
`^
VARIABLE_VALUEdownsampler_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdownsampler_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

m0
n1
 
²
Çlayers
Èlayer_metrics
o	variables
Énon_trainable_variables
Êmetrics
ptrainable_variables
 Ëlayer_regularization_losses
qregularization_losses
ig
VARIABLE_VALUEresblock_part1_1_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_1_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

s0
t1
 
²
Ìlayers
Ílayer_metrics
u	variables
Înon_trainable_variables
Ïmetrics
vtrainable_variables
 Ðlayer_regularization_losses
wregularization_losses
 
 
 
²
Ñlayers
Òlayer_metrics
y	variables
Ónon_trainable_variables
Ômetrics
ztrainable_variables
 Õlayer_regularization_losses
{regularization_losses
ig
VARIABLE_VALUEresblock_part1_1_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_1_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

}0
~1

}0
~1
 
´
Ölayers
×layer_metrics
	variables
Ønon_trainable_variables
Ùmetrics
trainable_variables
 Úlayer_regularization_losses
regularization_losses
 
 
ig
VARIABLE_VALUEresblock_part1_2_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_2_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
Ûlayers
Ülayer_metrics
	variables
Ýnon_trainable_variables
Þmetrics
trainable_variables
 ßlayer_regularization_losses
regularization_losses
 
 
 
µ
àlayers
álayer_metrics
	variables
ânon_trainable_variables
ãmetrics
trainable_variables
 älayer_regularization_losses
regularization_losses
ig
VARIABLE_VALUEresblock_part1_2_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_2_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
ålayers
ælayer_metrics
	variables
çnon_trainable_variables
èmetrics
trainable_variables
 élayer_regularization_losses
regularization_losses
 
 
ig
VARIABLE_VALUEresblock_part1_3_conv1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_3_conv1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
êlayers
ëlayer_metrics
	variables
ìnon_trainable_variables
ímetrics
trainable_variables
 îlayer_regularization_losses
regularization_losses
 
 
 
µ
ïlayers
ðlayer_metrics
	variables
ñnon_trainable_variables
òmetrics
trainable_variables
 ólayer_regularization_losses
regularization_losses
ig
VARIABLE_VALUEresblock_part1_3_conv2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_3_conv2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

¡0
¢1

¡0
¢1
 
µ
ôlayers
õlayer_metrics
£	variables
önon_trainable_variables
÷metrics
¤trainable_variables
 ølayer_regularization_losses
¥regularization_losses
 
 
ig
VARIABLE_VALUEresblock_part1_4_conv1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_4_conv1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

©0
ª1

©0
ª1
 
µ
ùlayers
úlayer_metrics
«	variables
ûnon_trainable_variables
ümetrics
¬trainable_variables
 ýlayer_regularization_losses
­regularization_losses
 
 
 
µ
þlayers
ÿlayer_metrics
¯	variables
non_trainable_variables
metrics
°trainable_variables
 layer_regularization_losses
±regularization_losses
ig
VARIABLE_VALUEresblock_part1_4_conv2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEresblock_part1_4_conv2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

³0
´1

³0
´1
 
µ
layers
layer_metrics
µ	variables
non_trainable_variables
metrics
¶trainable_variables
 layer_regularization_losses
·regularization_losses
 
 
 
 
 
µ
layers
layer_metrics
»	variables
non_trainable_variables
metrics
¼trainable_variables
 layer_regularization_losses
½regularization_losses
a_
VARIABLE_VALUEdownsampler_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdownsampler_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

¿0
À1

¿0
À1
 
µ
layers
layer_metrics
Á	variables
non_trainable_variables
metrics
Âtrainable_variables
 layer_regularization_losses
Ãregularization_losses
jh
VARIABLE_VALUEresblock_part2_1_conv1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_1_conv1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

Å0
Æ1

Å0
Æ1
 
µ
layers
layer_metrics
Ç	variables
non_trainable_variables
metrics
Ètrainable_variables
 layer_regularization_losses
Éregularization_losses
 
 
 
µ
layers
layer_metrics
Ë	variables
non_trainable_variables
metrics
Ìtrainable_variables
 layer_regularization_losses
Íregularization_losses
jh
VARIABLE_VALUEresblock_part2_1_conv2/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_1_conv2/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

Ï0
Ð1

Ï0
Ð1
 
µ
layers
layer_metrics
Ñ	variables
non_trainable_variables
metrics
Òtrainable_variables
  layer_regularization_losses
Óregularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_2_conv1/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_2_conv1/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

×0
Ø1

×0
Ø1
 
µ
¡layers
¢layer_metrics
Ù	variables
£non_trainable_variables
¤metrics
Útrainable_variables
 ¥layer_regularization_losses
Ûregularization_losses
 
 
 
µ
¦layers
§layer_metrics
Ý	variables
¨non_trainable_variables
©metrics
Þtrainable_variables
 ªlayer_regularization_losses
ßregularization_losses
jh
VARIABLE_VALUEresblock_part2_2_conv2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_2_conv2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

á0
â1

á0
â1
 
µ
«layers
¬layer_metrics
ã	variables
­non_trainable_variables
®metrics
ätrainable_variables
 ¯layer_regularization_losses
åregularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_3_conv1/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_3_conv1/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

é0
ê1

é0
ê1
 
µ
°layers
±layer_metrics
ë	variables
²non_trainable_variables
³metrics
ìtrainable_variables
 ´layer_regularization_losses
íregularization_losses
 
 
 
µ
µlayers
¶layer_metrics
ï	variables
·non_trainable_variables
¸metrics
ðtrainable_variables
 ¹layer_regularization_losses
ñregularization_losses
jh
VARIABLE_VALUEresblock_part2_3_conv2/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_3_conv2/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

ó0
ô1

ó0
ô1
 
µ
ºlayers
»layer_metrics
õ	variables
¼non_trainable_variables
½metrics
ötrainable_variables
 ¾layer_regularization_losses
÷regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_4_conv1/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_4_conv1/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

û0
ü1

û0
ü1
 
µ
¿layers
Àlayer_metrics
ý	variables
Ánon_trainable_variables
Âmetrics
þtrainable_variables
 Ãlayer_regularization_losses
ÿregularization_losses
 
 
 
µ
Älayers
Ålayer_metrics
	variables
Ænon_trainable_variables
Çmetrics
trainable_variables
 Èlayer_regularization_losses
regularization_losses
jh
VARIABLE_VALUEresblock_part2_4_conv2/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_4_conv2/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
Élayers
Êlayer_metrics
	variables
Ënon_trainable_variables
Ìmetrics
trainable_variables
 Ílayer_regularization_losses
regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_5_conv1/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_5_conv1/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
Îlayers
Ïlayer_metrics
	variables
Ðnon_trainable_variables
Ñmetrics
trainable_variables
 Òlayer_regularization_losses
regularization_losses
 
 
 
µ
Ólayers
Ôlayer_metrics
	variables
Õnon_trainable_variables
Ömetrics
trainable_variables
 ×layer_regularization_losses
regularization_losses
jh
VARIABLE_VALUEresblock_part2_5_conv2/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_5_conv2/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
Ølayers
Ùlayer_metrics
	variables
Únon_trainable_variables
Ûmetrics
trainable_variables
 Ülayer_regularization_losses
regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_6_conv1/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_6_conv1/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
µ
Ýlayers
Þlayer_metrics
¡	variables
ßnon_trainable_variables
àmetrics
¢trainable_variables
 álayer_regularization_losses
£regularization_losses
 
 
 
µ
âlayers
ãlayer_metrics
¥	variables
änon_trainable_variables
åmetrics
¦trainable_variables
 ælayer_regularization_losses
§regularization_losses
jh
VARIABLE_VALUEresblock_part2_6_conv2/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_6_conv2/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE

©0
ª1

©0
ª1
 
µ
çlayers
èlayer_metrics
«	variables
énon_trainable_variables
êmetrics
¬trainable_variables
 ëlayer_regularization_losses
­regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_7_conv1/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_7_conv1/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE

±0
²1

±0
²1
 
µ
ìlayers
ílayer_metrics
³	variables
înon_trainable_variables
ïmetrics
´trainable_variables
 ðlayer_regularization_losses
µregularization_losses
 
 
 
µ
ñlayers
òlayer_metrics
·	variables
ónon_trainable_variables
ômetrics
¸trainable_variables
 õlayer_regularization_losses
¹regularization_losses
jh
VARIABLE_VALUEresblock_part2_7_conv2/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_7_conv2/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE

»0
¼1

»0
¼1
 
µ
ölayers
÷layer_metrics
½	variables
ønon_trainable_variables
ùmetrics
¾trainable_variables
 úlayer_regularization_losses
¿regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part2_8_conv1/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_8_conv1/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE

Ã0
Ä1

Ã0
Ä1
 
µ
ûlayers
ülayer_metrics
Å	variables
ýnon_trainable_variables
þmetrics
Ætrainable_variables
 ÿlayer_regularization_losses
Çregularization_losses
 
 
 
µ
layers
layer_metrics
É	variables
non_trainable_variables
metrics
Êtrainable_variables
 layer_regularization_losses
Ëregularization_losses
jh
VARIABLE_VALUEresblock_part2_8_conv2/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part2_8_conv2/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE

Í0
Î1

Í0
Î1
 
µ
layers
layer_metrics
Ï	variables
non_trainable_variables
metrics
Ðtrainable_variables
 layer_regularization_losses
Ñregularization_losses
 
 
_]
VARIABLE_VALUEupsampler_1/kernel7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEupsampler_1/bias5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUE

Õ0
Ö1

Õ0
Ö1
 
µ
layers
layer_metrics
×	variables
non_trainable_variables
metrics
Øtrainable_variables
 layer_regularization_losses
Ùregularization_losses
 
jh
VARIABLE_VALUEresblock_part3_1_conv1/kernel7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_1_conv1/bias5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUE

Ü0
Ý1

Ü0
Ý1
 
µ
layers
layer_metrics
Þ	variables
non_trainable_variables
metrics
ßtrainable_variables
 layer_regularization_losses
àregularization_losses
 
 
 
µ
layers
layer_metrics
â	variables
non_trainable_variables
metrics
ãtrainable_variables
 layer_regularization_losses
äregularization_losses
jh
VARIABLE_VALUEresblock_part3_1_conv2/kernel7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_1_conv2/bias5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUE

æ0
ç1

æ0
ç1
 
µ
layers
layer_metrics
è	variables
non_trainable_variables
metrics
étrainable_variables
 layer_regularization_losses
êregularization_losses
 
 
jh
VARIABLE_VALUEresblock_part3_2_conv1/kernel7layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_2_conv1/bias5layer_with_weights-30/bias/.ATTRIBUTES/VARIABLE_VALUE

î0
ï1

î0
ï1
 
µ
layers
layer_metrics
ð	variables
 non_trainable_variables
¡metrics
ñtrainable_variables
 ¢layer_regularization_losses
òregularization_losses
 
 
 
µ
£layers
¤layer_metrics
ô	variables
¥non_trainable_variables
¦metrics
õtrainable_variables
 §layer_regularization_losses
öregularization_losses
jh
VARIABLE_VALUEresblock_part3_2_conv2/kernel7layer_with_weights-31/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_2_conv2/bias5layer_with_weights-31/bias/.ATTRIBUTES/VARIABLE_VALUE

ø0
ù1

ø0
ù1
 
µ
¨layers
©layer_metrics
ú	variables
ªnon_trainable_variables
«metrics
ûtrainable_variables
 ¬layer_regularization_losses
üregularization_losses
 
 
jh
VARIABLE_VALUEresblock_part3_3_conv1/kernel7layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_3_conv1/bias5layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
­layers
®layer_metrics
	variables
¯non_trainable_variables
°metrics
trainable_variables
 ±layer_regularization_losses
regularization_losses
 
 
 
µ
²layers
³layer_metrics
	variables
´non_trainable_variables
µmetrics
trainable_variables
 ¶layer_regularization_losses
regularization_losses
jh
VARIABLE_VALUEresblock_part3_3_conv2/kernel7layer_with_weights-33/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_3_conv2/bias5layer_with_weights-33/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
·layers
¸layer_metrics
	variables
¹non_trainable_variables
ºmetrics
trainable_variables
 »layer_regularization_losses
regularization_losses
 
 
jh
VARIABLE_VALUEresblock_part3_4_conv1/kernel7layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_4_conv1/bias5layer_with_weights-34/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
¼layers
½layer_metrics
	variables
¾non_trainable_variables
¿metrics
trainable_variables
 Àlayer_regularization_losses
regularization_losses
 
 
 
µ
Álayers
Âlayer_metrics
	variables
Ãnon_trainable_variables
Ämetrics
trainable_variables
 Ålayer_regularization_losses
regularization_losses
jh
VARIABLE_VALUEresblock_part3_4_conv2/kernel7layer_with_weights-35/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEresblock_part3_4_conv2/bias5layer_with_weights-35/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
Ælayers
Çlayer_metrics
	variables
Ènon_trainable_variables
Émetrics
trainable_variables
 Êlayer_regularization_losses
 regularization_losses
 
 
^\
VARIABLE_VALUEextra_conv/kernel7layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEextra_conv/bias5layer_with_weights-36/bias/.ATTRIBUTES/VARIABLE_VALUE

¤0
¥1

¤0
¥1
 
µ
Ëlayers
Ìlayer_metrics
¦	variables
Ínon_trainable_variables
Îmetrics
§trainable_variables
 Ïlayer_regularization_losses
¨regularization_losses
 
_]
VARIABLE_VALUEupsampler_2/kernel7layer_with_weights-37/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEupsampler_2/bias5layer_with_weights-37/bias/.ATTRIBUTES/VARIABLE_VALUE

«0
¬1

«0
¬1
 
µ
Ðlayers
Ñlayer_metrics
­	variables
Ònon_trainable_variables
Ómetrics
®trainable_variables
 Ôlayer_regularization_losses
¯regularization_losses
 
_]
VARIABLE_VALUEoutput_conv/kernel7layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_conv/bias5layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUE

²0
³1

²0
³1
 
µ
Õlayers
Ölayer_metrics
´	variables
×non_trainable_variables
Ømetrics
µtrainable_variables
 Ùlayer_regularization_losses
¶regularization_losses
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
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
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
"__inference_signature_wrapper_5646
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
__inference__traced_save_7824
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
 __inference__traced_restore_8068¤(
 

5__inference_resblock_part3_2_conv2_layer_call_fn_7398

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
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_40242
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
5__inference_resblock_part2_5_conv1_layer_call_fn_7110

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
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_36182
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
ä
f
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2974

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
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_3657

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
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_3793

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
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_3917

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
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4528
input_layer
input_conv_4263
input_conv_4265
downsampler_1_4269
downsampler_1_4271
resblock_part1_1_conv1_4274
resblock_part1_1_conv1_4276
resblock_part1_1_conv2_4280
resblock_part1_1_conv2_4282
tf_math_multiply_mul_x
resblock_part1_2_conv1_4288
resblock_part1_2_conv1_4290
resblock_part1_2_conv2_4294
resblock_part1_2_conv2_4296
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_4302
resblock_part1_3_conv1_4304
resblock_part1_3_conv2_4308
resblock_part1_3_conv2_4310
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_4316
resblock_part1_4_conv1_4318
resblock_part1_4_conv2_4322
resblock_part1_4_conv2_4324
tf_math_multiply_3_mul_x
downsampler_2_4331
downsampler_2_4333
resblock_part2_1_conv1_4336
resblock_part2_1_conv1_4338
resblock_part2_1_conv2_4342
resblock_part2_1_conv2_4344
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_4350
resblock_part2_2_conv1_4352
resblock_part2_2_conv2_4356
resblock_part2_2_conv2_4358
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_4364
resblock_part2_3_conv1_4366
resblock_part2_3_conv2_4370
resblock_part2_3_conv2_4372
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_4378
resblock_part2_4_conv1_4380
resblock_part2_4_conv2_4384
resblock_part2_4_conv2_4386
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_4392
resblock_part2_5_conv1_4394
resblock_part2_5_conv2_4398
resblock_part2_5_conv2_4400
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_4406
resblock_part2_6_conv1_4408
resblock_part2_6_conv2_4412
resblock_part2_6_conv2_4414
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_4420
resblock_part2_7_conv1_4422
resblock_part2_7_conv2_4426
resblock_part2_7_conv2_4428
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_4434
resblock_part2_8_conv1_4436
resblock_part2_8_conv2_4440
resblock_part2_8_conv2_4442
tf_math_multiply_11_mul_x
upsampler_1_4448
upsampler_1_4450
resblock_part3_1_conv1_4454
resblock_part3_1_conv1_4456
resblock_part3_1_conv2_4460
resblock_part3_1_conv2_4462
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_4468
resblock_part3_2_conv1_4470
resblock_part3_2_conv2_4474
resblock_part3_2_conv2_4476
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_4482
resblock_part3_3_conv1_4484
resblock_part3_3_conv2_4488
resblock_part3_3_conv2_4490
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_4496
resblock_part3_4_conv1_4498
resblock_part3_4_conv2_4502
resblock_part3_4_conv2_4504
tf_math_multiply_15_mul_x
extra_conv_4510
extra_conv_4512
upsampler_2_4516
upsampler_2_4518
output_conv_4522
output_conv_4524
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCallª
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinput_layerinput_conv_4263input_conv_4265*
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
D__inference_input_conv_layer_call_and_return_conditional_losses_29942$
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
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_29612 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_4269downsampler_1_4271*
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
G__inference_downsampler_1_layer_call_and_return_conditional_losses_30212'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_4274resblock_part1_1_conv1_4276*
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
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_304720
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
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_30682(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_4280resblock_part1_1_conv2_4282*
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
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_308620
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
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_4288resblock_part1_2_conv1_4290*
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
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_311520
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
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_31362(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_4294resblock_part1_2_conv2_4296*
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
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_315420
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
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_4302resblock_part1_3_conv1_4304*
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
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_318320
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
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_32042(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_4308resblock_part1_3_conv2_4310*
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
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_322220
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
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_4316resblock_part1_4_conv1_4318*
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
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_325120
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
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_32722(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_4322resblock_part1_4_conv2_4324*
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
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_329020
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
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_29742"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_4331downsampler_2_4333*
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
G__inference_downsampler_2_layer_call_and_return_conditional_losses_33202'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_4336resblock_part2_1_conv1_4338*
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
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_334620
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
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_33672(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_4342resblock_part2_1_conv2_4344*
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
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_338520
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
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_4350resblock_part2_2_conv1_4352*
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
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_341420
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
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_34352(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_4356resblock_part2_2_conv2_4358*
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
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_345320
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
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_4364resblock_part2_3_conv1_4366*
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
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_348220
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
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_35032(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_4370resblock_part2_3_conv2_4372*
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
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_352120
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
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_4378resblock_part2_4_conv1_4380*
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
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_355020
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
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_35712(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_4384resblock_part2_4_conv2_4386*
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
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_358920
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
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_4392resblock_part2_5_conv1_4394*
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
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_361820
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
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_36392(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_4398resblock_part2_5_conv2_4400*
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
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_365720
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
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_4406resblock_part2_6_conv1_4408*
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
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_368620
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
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_37072(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_4412resblock_part2_6_conv2_4414*
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
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_372520
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
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_4420resblock_part2_7_conv1_4422*
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
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_375420
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
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_37752(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_4426resblock_part2_7_conv2_4428*
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
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_379320
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
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_4434resblock_part2_8_conv1_4436*
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
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_382220
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
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_38432(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_4440resblock_part2_8_conv2_4442*
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
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_386120
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
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_4448upsampler_1_4450*
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
E__inference_upsampler_1_layer_call_and_return_conditional_losses_38902%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_4454resblock_part3_1_conv1_4456*
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
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_391720
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
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_39382(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_4460resblock_part3_1_conv2_4462*
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
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_395620
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
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_4468resblock_part3_2_conv1_4470*
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
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_398520
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
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_40062(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_4474resblock_part3_2_conv2_4476*
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
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_402420
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
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_4482resblock_part3_3_conv1_4484*
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
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_405320
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
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_40742(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_4488resblock_part3_3_conv2_4490*
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
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_409220
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
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_4496resblock_part3_4_conv1_4498*
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
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_412120
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
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_41422(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_4502resblock_part3_4_conv2_4504*
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
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_416020
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
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_4510extra_conv_4512*
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
D__inference_extra_conv_layer_call_and_return_conditional_losses_41892$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_4516upsampler_2_4518*
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
E__inference_upsampler_2_layer_call_and_return_conditional_losses_42162%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_4522output_conv_4524*
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
E__inference_output_conv_layer_call_and_return_conditional_losses_42432%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
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
:ÿÿÿÿÿÿÿÿÿ
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
æ
l
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_7470

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
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_4006

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
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_3222

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
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_6264

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
:@*
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
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2L
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
:ÿÿÿÿÿÿÿÿÿ
 
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
Þ
l
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_3639

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
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_3550

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
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_7101

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
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_6794

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
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_4024

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


,__inference_downsampler_1_layer_call_fn_6688

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
G__inference_downsampler_1_layer_call_and_return_conditional_losses_30212
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
å"
·
+__inference_ssi_res_unet_layer_call_fn_6650

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
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_52602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
¤

é
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_6909

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
D__inference_extra_conv_layer_call_and_return_conditional_losses_7504

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
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_3154

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
5__inference_resblock_part1_1_conv1_layer_call_fn_6707

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
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_30472
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
 

5__inference_resblock_part1_4_conv1_layer_call_fn_6851

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
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_32512
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
ª
I
-__inference_zero_padding2d_layer_call_fn_2967

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
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_29612
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


*__inference_upsampler_2_layer_call_fn_7532

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
E__inference_upsampler_2_layer_call_and_return_conditional_losses_42162
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
¨

Þ
E__inference_upsampler_2_layer_call_and_return_conditional_losses_7523

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
5__inference_resblock_part3_4_conv1_layer_call_fn_7465

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
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_41212
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
¦

à
G__inference_downsampler_1_layer_call_and_return_conditional_losses_6679

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
Ä"
³
"__inference_signature_wrapper_5646
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
__inference__wrapped_model_29542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
¢

Ý
D__inference_input_conv_layer_call_and_return_conditional_losses_6660

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
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
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_3346

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
5__inference_resblock_part2_4_conv1_layer_call_fn_7062

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
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_35502
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
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_7130

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
5__inference_resblock_part2_3_conv1_layer_call_fn_7014

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
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_34822
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
5__inference_resblock_part3_1_relu1_layer_call_fn_7331

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
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_39382
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
5__inference_resblock_part2_2_conv2_layer_call_fn_6995

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
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_34532
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
¢

Ý
D__inference_extra_conv_layer_call_and_return_conditional_losses_4189

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
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4799

inputs
input_conv_4534
input_conv_4536
downsampler_1_4540
downsampler_1_4542
resblock_part1_1_conv1_4545
resblock_part1_1_conv1_4547
resblock_part1_1_conv2_4551
resblock_part1_1_conv2_4553
tf_math_multiply_mul_x
resblock_part1_2_conv1_4559
resblock_part1_2_conv1_4561
resblock_part1_2_conv2_4565
resblock_part1_2_conv2_4567
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_4573
resblock_part1_3_conv1_4575
resblock_part1_3_conv2_4579
resblock_part1_3_conv2_4581
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_4587
resblock_part1_4_conv1_4589
resblock_part1_4_conv2_4593
resblock_part1_4_conv2_4595
tf_math_multiply_3_mul_x
downsampler_2_4602
downsampler_2_4604
resblock_part2_1_conv1_4607
resblock_part2_1_conv1_4609
resblock_part2_1_conv2_4613
resblock_part2_1_conv2_4615
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_4621
resblock_part2_2_conv1_4623
resblock_part2_2_conv2_4627
resblock_part2_2_conv2_4629
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_4635
resblock_part2_3_conv1_4637
resblock_part2_3_conv2_4641
resblock_part2_3_conv2_4643
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_4649
resblock_part2_4_conv1_4651
resblock_part2_4_conv2_4655
resblock_part2_4_conv2_4657
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_4663
resblock_part2_5_conv1_4665
resblock_part2_5_conv2_4669
resblock_part2_5_conv2_4671
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_4677
resblock_part2_6_conv1_4679
resblock_part2_6_conv2_4683
resblock_part2_6_conv2_4685
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_4691
resblock_part2_7_conv1_4693
resblock_part2_7_conv2_4697
resblock_part2_7_conv2_4699
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_4705
resblock_part2_8_conv1_4707
resblock_part2_8_conv2_4711
resblock_part2_8_conv2_4713
tf_math_multiply_11_mul_x
upsampler_1_4719
upsampler_1_4721
resblock_part3_1_conv1_4725
resblock_part3_1_conv1_4727
resblock_part3_1_conv2_4731
resblock_part3_1_conv2_4733
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_4739
resblock_part3_2_conv1_4741
resblock_part3_2_conv2_4745
resblock_part3_2_conv2_4747
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_4753
resblock_part3_3_conv1_4755
resblock_part3_3_conv2_4759
resblock_part3_3_conv2_4761
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_4767
resblock_part3_4_conv1_4769
resblock_part3_4_conv2_4773
resblock_part3_4_conv2_4775
tf_math_multiply_15_mul_x
extra_conv_4781
extra_conv_4783
upsampler_2_4787
upsampler_2_4789
output_conv_4793
output_conv_4795
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCall¥
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_4534input_conv_4536*
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
D__inference_input_conv_layer_call_and_return_conditional_losses_29942$
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
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_29612 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_4540downsampler_1_4542*
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
G__inference_downsampler_1_layer_call_and_return_conditional_losses_30212'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_4545resblock_part1_1_conv1_4547*
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
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_304720
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
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_30682(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_4551resblock_part1_1_conv2_4553*
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
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_308620
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
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_4559resblock_part1_2_conv1_4561*
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
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_311520
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
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_31362(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_4565resblock_part1_2_conv2_4567*
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
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_315420
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
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_4573resblock_part1_3_conv1_4575*
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
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_318320
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
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_32042(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_4579resblock_part1_3_conv2_4581*
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
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_322220
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
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_4587resblock_part1_4_conv1_4589*
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
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_325120
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
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_32722(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_4593resblock_part1_4_conv2_4595*
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
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_329020
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
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_29742"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_4602downsampler_2_4604*
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
G__inference_downsampler_2_layer_call_and_return_conditional_losses_33202'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_4607resblock_part2_1_conv1_4609*
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
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_334620
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
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_33672(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_4613resblock_part2_1_conv2_4615*
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
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_338520
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
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_4621resblock_part2_2_conv1_4623*
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
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_341420
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
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_34352(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_4627resblock_part2_2_conv2_4629*
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
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_345320
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
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_4635resblock_part2_3_conv1_4637*
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
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_348220
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
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_35032(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_4641resblock_part2_3_conv2_4643*
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
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_352120
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
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_4649resblock_part2_4_conv1_4651*
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
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_355020
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
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_35712(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_4655resblock_part2_4_conv2_4657*
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
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_358920
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
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_4663resblock_part2_5_conv1_4665*
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
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_361820
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
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_36392(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_4669resblock_part2_5_conv2_4671*
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
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_365720
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
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_4677resblock_part2_6_conv1_4679*
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
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_368620
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
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_37072(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_4683resblock_part2_6_conv2_4685*
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
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_372520
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
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_4691resblock_part2_7_conv1_4693*
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
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_375420
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
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_37752(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_4697resblock_part2_7_conv2_4699*
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
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_379320
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
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_4705resblock_part2_8_conv1_4707*
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
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_382220
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
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_38432(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_4711resblock_part2_8_conv2_4713*
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
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_386120
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
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_4719upsampler_1_4721*
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
E__inference_upsampler_1_layer_call_and_return_conditional_losses_38902%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_4725resblock_part3_1_conv1_4727*
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
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_391720
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
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_39382(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_4731resblock_part3_1_conv2_4733*
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
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_395620
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
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_4739resblock_part3_2_conv1_4741*
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
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_398520
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
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_40062(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_4745resblock_part3_2_conv2_4747*
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
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_402420
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
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_4753resblock_part3_3_conv1_4755*
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
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_405320
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
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_40742(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_4759resblock_part3_3_conv2_4761*
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
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_409220
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
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_4767resblock_part3_4_conv1_4769*
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
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_412120
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
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_41422(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_4773resblock_part3_4_conv2_4775*
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
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_416020
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
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_4781extra_conv_4783*
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
D__inference_extra_conv_layer_call_and_return_conditional_losses_41892$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_4787upsampler_2_4789*
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
E__inference_upsampler_2_layer_call_and_return_conditional_losses_42162%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_4793output_conv_4795*
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
E__inference_output_conv_layer_call_and_return_conditional_losses_42432%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
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
:ÿÿÿÿÿÿÿÿÿ
 
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
Õ
Q
5__inference_resblock_part1_4_relu1_layer_call_fn_6861

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
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_32722
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
Í
Q
5__inference_resblock_part2_6_relu1_layer_call_fn_7168

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
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_37072
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
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_3861

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
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_7408

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


*__inference_output_conv_layer_call_fn_7551

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
E__inference_output_conv_layer_call_and_return_conditional_losses_42432
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
Þ
l
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_7067

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
5__inference_resblock_part3_3_conv1_layer_call_fn_7417

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
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_40532
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
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_4142

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
â
d
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_2961

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
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_3589

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
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_6775

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
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_6712

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
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_7274

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
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_3047

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
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_3571

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
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_3618

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
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_3136

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
*__inference_upsampler_1_layer_call_fn_7302

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
E__inference_upsampler_1_layer_call_and_return_conditional_losses_38902
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


5__inference_resblock_part2_7_conv1_layer_call_fn_7206

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
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_37542
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
æ
l
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_3204

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
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_6698

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
5__inference_resblock_part1_2_conv1_layer_call_fn_6755

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
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_31152
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
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_3385

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
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_3956

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
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_6760

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


Þ
E__inference_upsampler_1_layer_call_and_return_conditional_losses_7293

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
®

é
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_6842

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
5__inference_resblock_part1_3_conv1_layer_call_fn_6803

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
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_31832
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
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_7178

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
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_7019

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
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_3707

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
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_6986

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
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5955

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
:@*
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
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2L
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
:ÿÿÿÿÿÿÿÿÿ
 
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
¤

é
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_7053

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

~
)__inference_extra_conv_layer_call_fn_7513

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
D__inference_extra_conv_layer_call_and_return_conditional_losses_41892
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
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_3414

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
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_3482

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
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_7163

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
5__inference_resblock_part2_4_conv2_layer_call_fn_7091

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
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_35892
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
æ
l
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_3068

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
5__inference_resblock_part1_1_conv2_layer_call_fn_6736

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
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_30862
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
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_3183

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
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_3503

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
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_3086

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
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_4121

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
5__inference_resblock_part2_3_relu1_layer_call_fn_7024

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
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_35032
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
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_6923

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
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_3754

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
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_6971

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
5__inference_resblock_part2_7_conv2_layer_call_fn_7235

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
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_37932
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
æ
l
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_6808

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

~
)__inference_input_conv_layer_call_fn_6669

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
D__inference_input_conv_layer_call_and_return_conditional_losses_29942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

é
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_3521

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
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_3251

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
5__inference_resblock_part2_1_relu1_layer_call_fn_6928

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
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_33672
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
5__inference_resblock_part2_8_conv1_layer_call_fn_7254

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
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_38222
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
5__inference_resblock_part2_6_conv1_layer_call_fn_7158

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
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_36862
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
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_3115

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
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_7389

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
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_7211

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
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_7360

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
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_7082

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
®
K
/__inference_zero_padding2d_1_layer_call_fn_2980

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
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_29742
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
Í
Q
5__inference_resblock_part2_2_relu1_layer_call_fn_6976

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
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_34352
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
Õ
Q
5__inference_resblock_part3_2_relu1_layer_call_fn_7379

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
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_40062
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
Í
Q
5__inference_resblock_part2_5_relu1_layer_call_fn_7120

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
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_36392
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
æ
l
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_7374

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
5__inference_resblock_part3_3_conv2_layer_call_fn_7446

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
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_40922
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
E__inference_output_conv_layer_call_and_return_conditional_losses_7542

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


,__inference_downsampler_2_layer_call_fn_6899

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
G__inference_downsampler_2_layer_call_and_return_conditional_losses_33202
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
ô"
¼
+__inference_ssi_res_unet_layer_call_fn_5451
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
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_52602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
æ
l
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_7326

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
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_3938

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
îÆ
Á-
 __inference__traced_restore_8068
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


5__inference_resblock_part2_6_conv2_layer_call_fn_7187

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
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_37252
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
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_3453

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
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_7259

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
¦

à
G__inference_downsampler_1_layer_call_and_return_conditional_losses_3021

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
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_7005

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
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_6871

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
5__inference_resblock_part3_4_conv2_layer_call_fn_7494

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
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_41602
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
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_6856

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
é
í%
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5260

inputs
input_conv_4995
input_conv_4997
downsampler_1_5001
downsampler_1_5003
resblock_part1_1_conv1_5006
resblock_part1_1_conv1_5008
resblock_part1_1_conv2_5012
resblock_part1_1_conv2_5014
tf_math_multiply_mul_x
resblock_part1_2_conv1_5020
resblock_part1_2_conv1_5022
resblock_part1_2_conv2_5026
resblock_part1_2_conv2_5028
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_5034
resblock_part1_3_conv1_5036
resblock_part1_3_conv2_5040
resblock_part1_3_conv2_5042
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_5048
resblock_part1_4_conv1_5050
resblock_part1_4_conv2_5054
resblock_part1_4_conv2_5056
tf_math_multiply_3_mul_x
downsampler_2_5063
downsampler_2_5065
resblock_part2_1_conv1_5068
resblock_part2_1_conv1_5070
resblock_part2_1_conv2_5074
resblock_part2_1_conv2_5076
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_5082
resblock_part2_2_conv1_5084
resblock_part2_2_conv2_5088
resblock_part2_2_conv2_5090
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_5096
resblock_part2_3_conv1_5098
resblock_part2_3_conv2_5102
resblock_part2_3_conv2_5104
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_5110
resblock_part2_4_conv1_5112
resblock_part2_4_conv2_5116
resblock_part2_4_conv2_5118
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_5124
resblock_part2_5_conv1_5126
resblock_part2_5_conv2_5130
resblock_part2_5_conv2_5132
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_5138
resblock_part2_6_conv1_5140
resblock_part2_6_conv2_5144
resblock_part2_6_conv2_5146
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_5152
resblock_part2_7_conv1_5154
resblock_part2_7_conv2_5158
resblock_part2_7_conv2_5160
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_5166
resblock_part2_8_conv1_5168
resblock_part2_8_conv2_5172
resblock_part2_8_conv2_5174
tf_math_multiply_11_mul_x
upsampler_1_5180
upsampler_1_5182
resblock_part3_1_conv1_5186
resblock_part3_1_conv1_5188
resblock_part3_1_conv2_5192
resblock_part3_1_conv2_5194
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_5200
resblock_part3_2_conv1_5202
resblock_part3_2_conv2_5206
resblock_part3_2_conv2_5208
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_5214
resblock_part3_3_conv1_5216
resblock_part3_3_conv2_5220
resblock_part3_3_conv2_5222
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_5228
resblock_part3_4_conv1_5230
resblock_part3_4_conv2_5234
resblock_part3_4_conv2_5236
tf_math_multiply_15_mul_x
extra_conv_5242
extra_conv_5244
upsampler_2_5248
upsampler_2_5250
output_conv_5254
output_conv_5256
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCall¥
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_4995input_conv_4997*
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
D__inference_input_conv_layer_call_and_return_conditional_losses_29942$
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
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_29612 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_5001downsampler_1_5003*
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
G__inference_downsampler_1_layer_call_and_return_conditional_losses_30212'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_5006resblock_part1_1_conv1_5008*
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
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_304720
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
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_30682(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_5012resblock_part1_1_conv2_5014*
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
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_308620
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
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_5020resblock_part1_2_conv1_5022*
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
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_311520
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
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_31362(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_5026resblock_part1_2_conv2_5028*
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
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_315420
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
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_5034resblock_part1_3_conv1_5036*
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
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_318320
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
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_32042(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_5040resblock_part1_3_conv2_5042*
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
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_322220
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
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_5048resblock_part1_4_conv1_5050*
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
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_325120
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
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_32722(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_5054resblock_part1_4_conv2_5056*
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
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_329020
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
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_29742"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_5063downsampler_2_5065*
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
G__inference_downsampler_2_layer_call_and_return_conditional_losses_33202'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_5068resblock_part2_1_conv1_5070*
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
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_334620
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
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_33672(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_5074resblock_part2_1_conv2_5076*
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
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_338520
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
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_5082resblock_part2_2_conv1_5084*
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
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_341420
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
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_34352(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_5088resblock_part2_2_conv2_5090*
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
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_345320
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
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_5096resblock_part2_3_conv1_5098*
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
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_348220
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
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_35032(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_5102resblock_part2_3_conv2_5104*
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
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_352120
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
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_5110resblock_part2_4_conv1_5112*
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
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_355020
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
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_35712(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_5116resblock_part2_4_conv2_5118*
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
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_358920
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
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_5124resblock_part2_5_conv1_5126*
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
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_361820
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
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_36392(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_5130resblock_part2_5_conv2_5132*
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
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_365720
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
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_5138resblock_part2_6_conv1_5140*
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
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_368620
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
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_37072(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_5144resblock_part2_6_conv2_5146*
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
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_372520
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
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_5152resblock_part2_7_conv1_5154*
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
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_375420
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
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_37752(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_5158resblock_part2_7_conv2_5160*
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
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_379320
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
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_5166resblock_part2_8_conv1_5168*
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
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_382220
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
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_38432(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_5172resblock_part2_8_conv2_5174*
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
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_386120
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
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_5180upsampler_1_5182*
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
E__inference_upsampler_1_layer_call_and_return_conditional_losses_38902%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_5186resblock_part3_1_conv1_5188*
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
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_391720
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
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_39382(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_5192resblock_part3_1_conv2_5194*
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
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_395620
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
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_5200resblock_part3_2_conv1_5202*
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
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_398520
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
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_40062(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_5206resblock_part3_2_conv2_5208*
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
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_402420
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
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_5214resblock_part3_3_conv1_5216*
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
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_405320
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
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_40742(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_5220resblock_part3_3_conv2_5222*
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
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_409220
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
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_5228resblock_part3_4_conv1_5230*
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
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_412120
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
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_41422(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_5234resblock_part3_4_conv2_5236*
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
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_416020
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
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_5242extra_conv_5244*
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
D__inference_extra_conv_layer_call_and_return_conditional_losses_41892$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_5248upsampler_2_5250*
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
E__inference_upsampler_2_layer_call_and_return_conditional_losses_42162%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_5254output_conv_5256*
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
E__inference_output_conv_layer_call_and_return_conditional_losses_42432%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
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
:ÿÿÿÿÿÿÿÿÿ
 
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
¤

é
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_3822

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
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_7115

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
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_7312

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
5__inference_resblock_part3_2_conv1_layer_call_fn_7369

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
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_39852
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
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_6727

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
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_6746

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
5__inference_resblock_part1_3_conv2_layer_call_fn_6832

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
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_32222
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
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_7034

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
¨

Þ
E__inference_upsampler_2_layer_call_and_return_conditional_losses_4216

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
 

à
G__inference_downsampler_2_layer_call_and_return_conditional_losses_3320

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
Í
Q
5__inference_resblock_part2_7_relu1_layer_call_fn_7216

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
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_37752
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
®

é
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_4160

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
5__inference_resblock_part1_1_relu1_layer_call_fn_6717

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
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_30682
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
5__inference_resblock_part2_5_conv2_layer_call_fn_7139

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
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_36572
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
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_3686

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
5__inference_resblock_part1_4_conv2_layer_call_fn_6880

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
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_32902
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
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_7226

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
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_4074

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
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_7485

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
5__inference_resblock_part1_2_relu1_layer_call_fn_6765

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
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_31362
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
5__inference_resblock_part2_3_conv2_layer_call_fn_7043

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
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_35212
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
5__inference_resblock_part3_4_relu1_layer_call_fn_7475

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
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_41422
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
 

5__inference_resblock_part3_1_conv1_layer_call_fn_7321

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
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_39172
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
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_3290

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
5__inference_resblock_part2_1_conv1_layer_call_fn_6918

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
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_33462
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
æ
l
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_3272

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
ô"
¼
+__inference_ssi_res_unet_layer_call_fn_4990
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
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_47992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
ñ
í$
__inference__traced_save_7824
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
°: :@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 
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
 

à
G__inference_downsampler_2_layer_call_and_return_conditional_losses_6890

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


5__inference_resblock_part2_2_conv1_layer_call_fn_6966

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
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_34142
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
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_6823

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
D__inference_input_conv_layer_call_and_return_conditional_losses_2994

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
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
%:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
ÑT
__inference__wrapped_model_2954
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
:@*
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
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2f
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
:ÿÿÿÿÿÿÿÿÿ
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
 

5__inference_resblock_part1_2_conv2_layer_call_fn_6784

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
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_31542
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
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_7245

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
5__inference_resblock_part2_1_conv2_layer_call_fn_6947

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
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_33852
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
æ
l
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_7422

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
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_4053

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
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_7341

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
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4260
input_layer
input_conv_3005
input_conv_3007
downsampler_1_3032
downsampler_1_3034
resblock_part1_1_conv1_3058
resblock_part1_1_conv1_3060
resblock_part1_1_conv2_3097
resblock_part1_1_conv2_3099
tf_math_multiply_mul_x
resblock_part1_2_conv1_3126
resblock_part1_2_conv1_3128
resblock_part1_2_conv2_3165
resblock_part1_2_conv2_3167
tf_math_multiply_1_mul_x
resblock_part1_3_conv1_3194
resblock_part1_3_conv1_3196
resblock_part1_3_conv2_3233
resblock_part1_3_conv2_3235
tf_math_multiply_2_mul_x
resblock_part1_4_conv1_3262
resblock_part1_4_conv1_3264
resblock_part1_4_conv2_3301
resblock_part1_4_conv2_3303
tf_math_multiply_3_mul_x
downsampler_2_3331
downsampler_2_3333
resblock_part2_1_conv1_3357
resblock_part2_1_conv1_3359
resblock_part2_1_conv2_3396
resblock_part2_1_conv2_3398
tf_math_multiply_4_mul_x
resblock_part2_2_conv1_3425
resblock_part2_2_conv1_3427
resblock_part2_2_conv2_3464
resblock_part2_2_conv2_3466
tf_math_multiply_5_mul_x
resblock_part2_3_conv1_3493
resblock_part2_3_conv1_3495
resblock_part2_3_conv2_3532
resblock_part2_3_conv2_3534
tf_math_multiply_6_mul_x
resblock_part2_4_conv1_3561
resblock_part2_4_conv1_3563
resblock_part2_4_conv2_3600
resblock_part2_4_conv2_3602
tf_math_multiply_7_mul_x
resblock_part2_5_conv1_3629
resblock_part2_5_conv1_3631
resblock_part2_5_conv2_3668
resblock_part2_5_conv2_3670
tf_math_multiply_8_mul_x
resblock_part2_6_conv1_3697
resblock_part2_6_conv1_3699
resblock_part2_6_conv2_3736
resblock_part2_6_conv2_3738
tf_math_multiply_9_mul_x
resblock_part2_7_conv1_3765
resblock_part2_7_conv1_3767
resblock_part2_7_conv2_3804
resblock_part2_7_conv2_3806
tf_math_multiply_10_mul_x
resblock_part2_8_conv1_3833
resblock_part2_8_conv1_3835
resblock_part2_8_conv2_3872
resblock_part2_8_conv2_3874
tf_math_multiply_11_mul_x
upsampler_1_3901
upsampler_1_3903
resblock_part3_1_conv1_3928
resblock_part3_1_conv1_3930
resblock_part3_1_conv2_3967
resblock_part3_1_conv2_3969
tf_math_multiply_12_mul_x
resblock_part3_2_conv1_3996
resblock_part3_2_conv1_3998
resblock_part3_2_conv2_4035
resblock_part3_2_conv2_4037
tf_math_multiply_13_mul_x
resblock_part3_3_conv1_4064
resblock_part3_3_conv1_4066
resblock_part3_3_conv2_4103
resblock_part3_3_conv2_4105
tf_math_multiply_14_mul_x
resblock_part3_4_conv1_4132
resblock_part3_4_conv1_4134
resblock_part3_4_conv2_4171
resblock_part3_4_conv2_4173
tf_math_multiply_15_mul_x
extra_conv_4200
extra_conv_4202
upsampler_2_4227
upsampler_2_4229
output_conv_4254
output_conv_4256
identity¢%downsampler_1/StatefulPartitionedCall¢%downsampler_2/StatefulPartitionedCall¢"extra_conv/StatefulPartitionedCall¢"input_conv/StatefulPartitionedCall¢#output_conv/StatefulPartitionedCall¢.resblock_part1_1_conv1/StatefulPartitionedCall¢.resblock_part1_1_conv2/StatefulPartitionedCall¢.resblock_part1_2_conv1/StatefulPartitionedCall¢.resblock_part1_2_conv2/StatefulPartitionedCall¢.resblock_part1_3_conv1/StatefulPartitionedCall¢.resblock_part1_3_conv2/StatefulPartitionedCall¢.resblock_part1_4_conv1/StatefulPartitionedCall¢.resblock_part1_4_conv2/StatefulPartitionedCall¢.resblock_part2_1_conv1/StatefulPartitionedCall¢.resblock_part2_1_conv2/StatefulPartitionedCall¢.resblock_part2_2_conv1/StatefulPartitionedCall¢.resblock_part2_2_conv2/StatefulPartitionedCall¢.resblock_part2_3_conv1/StatefulPartitionedCall¢.resblock_part2_3_conv2/StatefulPartitionedCall¢.resblock_part2_4_conv1/StatefulPartitionedCall¢.resblock_part2_4_conv2/StatefulPartitionedCall¢.resblock_part2_5_conv1/StatefulPartitionedCall¢.resblock_part2_5_conv2/StatefulPartitionedCall¢.resblock_part2_6_conv1/StatefulPartitionedCall¢.resblock_part2_6_conv2/StatefulPartitionedCall¢.resblock_part2_7_conv1/StatefulPartitionedCall¢.resblock_part2_7_conv2/StatefulPartitionedCall¢.resblock_part2_8_conv1/StatefulPartitionedCall¢.resblock_part2_8_conv2/StatefulPartitionedCall¢.resblock_part3_1_conv1/StatefulPartitionedCall¢.resblock_part3_1_conv2/StatefulPartitionedCall¢.resblock_part3_2_conv1/StatefulPartitionedCall¢.resblock_part3_2_conv2/StatefulPartitionedCall¢.resblock_part3_3_conv1/StatefulPartitionedCall¢.resblock_part3_3_conv2/StatefulPartitionedCall¢.resblock_part3_4_conv1/StatefulPartitionedCall¢.resblock_part3_4_conv2/StatefulPartitionedCall¢#upsampler_1/StatefulPartitionedCall¢#upsampler_2/StatefulPartitionedCallª
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinput_layerinput_conv_3005input_conv_3007*
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
D__inference_input_conv_layer_call_and_return_conditional_losses_29942$
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
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_29612 
zero_padding2d/PartitionedCallÕ
%downsampler_1/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0downsampler_1_3032downsampler_1_3034*
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
G__inference_downsampler_1_layer_call_and_return_conditional_losses_30212'
%downsampler_1/StatefulPartitionedCall
.resblock_part1_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_1/StatefulPartitionedCall:output:0resblock_part1_1_conv1_3058resblock_part1_1_conv1_3060*
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
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_304720
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
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_30682(
&resblock_part1_1_relu1/PartitionedCall
.resblock_part1_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_1_relu1/PartitionedCall:output:0resblock_part1_1_conv2_3097resblock_part1_1_conv2_3099*
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
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_308620
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
.resblock_part1_2_conv1/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0resblock_part1_2_conv1_3126resblock_part1_2_conv1_3128*
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
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_311520
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
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_31362(
&resblock_part1_2_relu1/PartitionedCall
.resblock_part1_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_2_relu1/PartitionedCall:output:0resblock_part1_2_conv2_3165resblock_part1_2_conv2_3167*
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
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_315420
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
.resblock_part1_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0resblock_part1_3_conv1_3194resblock_part1_3_conv1_3196*
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
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_318320
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
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_32042(
&resblock_part1_3_relu1/PartitionedCall
.resblock_part1_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_3_relu1/PartitionedCall:output:0resblock_part1_3_conv2_3233resblock_part1_3_conv2_3235*
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
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_322220
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
.resblock_part1_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0resblock_part1_4_conv1_3262resblock_part1_4_conv1_3264*
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
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_325120
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
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_32722(
&resblock_part1_4_relu1/PartitionedCall
.resblock_part1_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part1_4_relu1/PartitionedCall:output:0resblock_part1_4_conv2_3301resblock_part1_4_conv2_3303*
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
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_329020
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
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_29742"
 zero_padding2d_1/PartitionedCallÕ
%downsampler_2/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_1/PartitionedCall:output:0downsampler_2_3331downsampler_2_3333*
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
G__inference_downsampler_2_layer_call_and_return_conditional_losses_33202'
%downsampler_2/StatefulPartitionedCall
.resblock_part2_1_conv1/StatefulPartitionedCallStatefulPartitionedCall.downsampler_2/StatefulPartitionedCall:output:0resblock_part2_1_conv1_3357resblock_part2_1_conv1_3359*
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
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_334620
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
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_33672(
&resblock_part2_1_relu1/PartitionedCall
.resblock_part2_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_1_relu1/PartitionedCall:output:0resblock_part2_1_conv2_3396resblock_part2_1_conv2_3398*
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
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_338520
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
.resblock_part2_2_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0resblock_part2_2_conv1_3425resblock_part2_2_conv1_3427*
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
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_341420
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
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_34352(
&resblock_part2_2_relu1/PartitionedCall
.resblock_part2_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_2_relu1/PartitionedCall:output:0resblock_part2_2_conv2_3464resblock_part2_2_conv2_3466*
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
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_345320
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
.resblock_part2_3_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0resblock_part2_3_conv1_3493resblock_part2_3_conv1_3495*
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
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_348220
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
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_35032(
&resblock_part2_3_relu1/PartitionedCall
.resblock_part2_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_3_relu1/PartitionedCall:output:0resblock_part2_3_conv2_3532resblock_part2_3_conv2_3534*
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
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_352120
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
.resblock_part2_4_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_6/AddV2:z:0resblock_part2_4_conv1_3561resblock_part2_4_conv1_3563*
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
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_355020
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
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_35712(
&resblock_part2_4_relu1/PartitionedCall
.resblock_part2_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_4_relu1/PartitionedCall:output:0resblock_part2_4_conv2_3600resblock_part2_4_conv2_3602*
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
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_358920
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
.resblock_part2_5_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_7/AddV2:z:0resblock_part2_5_conv1_3629resblock_part2_5_conv1_3631*
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
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_361820
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
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_36392(
&resblock_part2_5_relu1/PartitionedCall
.resblock_part2_5_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_5_relu1/PartitionedCall:output:0resblock_part2_5_conv2_3668resblock_part2_5_conv2_3670*
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
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_365720
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
.resblock_part2_6_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_8/AddV2:z:0resblock_part2_6_conv1_3697resblock_part2_6_conv1_3699*
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
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_368620
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
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_37072(
&resblock_part2_6_relu1/PartitionedCall
.resblock_part2_6_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_6_relu1/PartitionedCall:output:0resblock_part2_6_conv2_3736resblock_part2_6_conv2_3738*
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
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_372520
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
.resblock_part2_7_conv1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_9/AddV2:z:0resblock_part2_7_conv1_3765resblock_part2_7_conv1_3767*
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
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_375420
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
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_37752(
&resblock_part2_7_relu1/PartitionedCall
.resblock_part2_7_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_7_relu1/PartitionedCall:output:0resblock_part2_7_conv2_3804resblock_part2_7_conv2_3806*
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
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_379320
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
.resblock_part2_8_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_10/AddV2:z:0resblock_part2_8_conv1_3833resblock_part2_8_conv1_3835*
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
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_382220
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
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_38432(
&resblock_part2_8_relu1/PartitionedCall
.resblock_part2_8_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part2_8_relu1/PartitionedCall:output:0resblock_part2_8_conv2_3872resblock_part2_8_conv2_3874*
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
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_386120
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
#upsampler_1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_11/AddV2:z:0upsampler_1_3901upsampler_1_3903*
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
E__inference_upsampler_1_layer_call_and_return_conditional_losses_38902%
#upsampler_1/StatefulPartitionedCallé
!tf.nn.depth_to_space/DepthToSpaceDepthToSpace,upsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2#
!tf.nn.depth_to_space/DepthToSpace
.resblock_part3_1_conv1/StatefulPartitionedCallStatefulPartitionedCall*tf.nn.depth_to_space/DepthToSpace:output:0resblock_part3_1_conv1_3928resblock_part3_1_conv1_3930*
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
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_391720
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
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_39382(
&resblock_part3_1_relu1/PartitionedCall
.resblock_part3_1_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_1_relu1/PartitionedCall:output:0resblock_part3_1_conv2_3967resblock_part3_1_conv2_3969*
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
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_395620
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
.resblock_part3_2_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_12/AddV2:z:0resblock_part3_2_conv1_3996resblock_part3_2_conv1_3998*
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
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_398520
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
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_40062(
&resblock_part3_2_relu1/PartitionedCall
.resblock_part3_2_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_2_relu1/PartitionedCall:output:0resblock_part3_2_conv2_4035resblock_part3_2_conv2_4037*
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
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_402420
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
.resblock_part3_3_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_13/AddV2:z:0resblock_part3_3_conv1_4064resblock_part3_3_conv1_4066*
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
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_405320
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
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_40742(
&resblock_part3_3_relu1/PartitionedCall
.resblock_part3_3_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_3_relu1/PartitionedCall:output:0resblock_part3_3_conv2_4103resblock_part3_3_conv2_4105*
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
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_409220
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
.resblock_part3_4_conv1/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_14/AddV2:z:0resblock_part3_4_conv1_4132resblock_part3_4_conv1_4134*
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
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_412120
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
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_41422(
&resblock_part3_4_relu1/PartitionedCall
.resblock_part3_4_conv2/StatefulPartitionedCallStatefulPartitionedCall/resblock_part3_4_relu1/PartitionedCall:output:0resblock_part3_4_conv2_4171resblock_part3_4_conv2_4173*
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
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_416020
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
"extra_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_15/AddV2:z:0extra_conv_4200extra_conv_4202*
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
D__inference_extra_conv_layer_call_and_return_conditional_losses_41892$
"extra_conv/StatefulPartitionedCallà
tf.__operators__.add_16/AddV2AddV2+extra_conv/StatefulPartitionedCall:output:0.downsampler_1/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
tf.__operators__.add_16/AddV2Æ
#upsampler_2/StatefulPartitionedCallStatefulPartitionedCall!tf.__operators__.add_16/AddV2:z:0upsampler_2_4227upsampler_2_4229*
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
E__inference_upsampler_2_layer_call_and_return_conditional_losses_42162%
#upsampler_2/StatefulPartitionedCallí
#tf.nn.depth_to_space_1/DepthToSpaceDepthToSpace,upsampler_2/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*

block_size*
data_formatNCHW2%
#tf.nn.depth_to_space_1/DepthToSpaceÐ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall,tf.nn.depth_to_space_1/DepthToSpace:output:0output_conv_4254output_conv_4256*
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
E__inference_output_conv_layer_call_and_return_conditional_losses_42432%
#output_conv/StatefulPartitionedCall¶
IdentityIdentity,output_conv/StatefulPartitionedCall:output:0&^downsampler_1/StatefulPartitionedCall&^downsampler_2/StatefulPartitionedCall#^extra_conv/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall/^resblock_part1_1_conv1/StatefulPartitionedCall/^resblock_part1_1_conv2/StatefulPartitionedCall/^resblock_part1_2_conv1/StatefulPartitionedCall/^resblock_part1_2_conv2/StatefulPartitionedCall/^resblock_part1_3_conv1/StatefulPartitionedCall/^resblock_part1_3_conv2/StatefulPartitionedCall/^resblock_part1_4_conv1/StatefulPartitionedCall/^resblock_part1_4_conv2/StatefulPartitionedCall/^resblock_part2_1_conv1/StatefulPartitionedCall/^resblock_part2_1_conv2/StatefulPartitionedCall/^resblock_part2_2_conv1/StatefulPartitionedCall/^resblock_part2_2_conv2/StatefulPartitionedCall/^resblock_part2_3_conv1/StatefulPartitionedCall/^resblock_part2_3_conv2/StatefulPartitionedCall/^resblock_part2_4_conv1/StatefulPartitionedCall/^resblock_part2_4_conv2/StatefulPartitionedCall/^resblock_part2_5_conv1/StatefulPartitionedCall/^resblock_part2_5_conv2/StatefulPartitionedCall/^resblock_part2_6_conv1/StatefulPartitionedCall/^resblock_part2_6_conv2/StatefulPartitionedCall/^resblock_part2_7_conv1/StatefulPartitionedCall/^resblock_part2_7_conv2/StatefulPartitionedCall/^resblock_part2_8_conv1/StatefulPartitionedCall/^resblock_part2_8_conv2/StatefulPartitionedCall/^resblock_part3_1_conv1/StatefulPartitionedCall/^resblock_part3_1_conv2/StatefulPartitionedCall/^resblock_part3_2_conv1/StatefulPartitionedCall/^resblock_part3_2_conv2/StatefulPartitionedCall/^resblock_part3_3_conv1/StatefulPartitionedCall/^resblock_part3_3_conv2/StatefulPartitionedCall/^resblock_part3_4_conv1/StatefulPartitionedCall/^resblock_part3_4_conv2/StatefulPartitionedCall$^upsampler_1/StatefulPartitionedCall$^upsampler_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::2N
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
:ÿÿÿÿÿÿÿÿÿ
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
5__inference_resblock_part2_4_relu1_layer_call_fn_7072

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
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_35712
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
®

é
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_3985

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
5__inference_resblock_part1_3_relu1_layer_call_fn_6813

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
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_32042
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
¤

é
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_6957

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
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_6938

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
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_3843

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


Þ
E__inference_upsampler_1_layer_call_and_return_conditional_losses_3890

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
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_3435

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
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_3725

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
5__inference_resblock_part3_3_relu1_layer_call_fn_7427

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
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_40742
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
£

Þ
E__inference_output_conv_layer_call_and_return_conditional_losses_4243

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
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_7149

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
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_7197

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
5__inference_resblock_part2_8_relu1_layer_call_fn_7264

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
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_38432
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
®

é
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_7437

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
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_3775

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
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_4092

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
5__inference_resblock_part3_1_conv2_layer_call_fn_7350

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
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_39562
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
å"
·
+__inference_ssi_res_unet_layer_call_fn_6457

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
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_47992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesø
õ:ÿÿÿÿÿÿÿÿÿ::::::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::: ::::::: ::::: ::::: ::::: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
®

é
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_7456

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
5__inference_resblock_part2_8_conv2_layer_call_fn_7283

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
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_38612
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
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_3367

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
serving_default_input_layer:0ÿÿÿÿÿÿÿÿÿI
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
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b
signatures
Ú_default_save_signature
Û__call__
+Ü&call_and_return_all_conditional_losses"´Ê
_tf_keras_networkÊ{"class_name": "Functional", "name": "ssi_res_unet", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "ssi_res_unet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 256, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "input_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "input_conv", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d", "inbound_nodes": [[["input_conv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_1", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv1", "inbound_nodes": [[["downsampler_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_1_relu1", "inbound_nodes": [[["resblock_part1_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv2", "inbound_nodes": [[["resblock_part1_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["tf.math.multiply", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv1", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_2_relu1", "inbound_nodes": [[["resblock_part1_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv2", "inbound_nodes": [[["resblock_part1_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"y": ["tf.__operators__.add", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_3_relu1", "inbound_nodes": [[["resblock_part1_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv2", "inbound_nodes": [[["resblock_part1_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"y": ["tf.__operators__.add_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv1", "inbound_nodes": [[["tf.__operators__.add_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_4_relu1", "inbound_nodes": [[["resblock_part1_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv2", "inbound_nodes": [[["resblock_part1_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"y": ["tf.__operators__.add_2", 0, 0], "name": null}]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d_1", "inbound_nodes": [[["tf.__operators__.add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_2", "inbound_nodes": [[["zero_padding2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv1", "inbound_nodes": [[["downsampler_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_1_relu1", "inbound_nodes": [[["resblock_part2_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv2", "inbound_nodes": [[["resblock_part2_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"y": ["downsampler_2", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv1", "inbound_nodes": [[["tf.__operators__.add_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_2_relu1", "inbound_nodes": [[["resblock_part2_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv2", "inbound_nodes": [[["resblock_part2_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"y": ["tf.__operators__.add_4", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv1", "inbound_nodes": [[["tf.__operators__.add_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_3_relu1", "inbound_nodes": [[["resblock_part2_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv2", "inbound_nodes": [[["resblock_part2_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"y": ["tf.__operators__.add_5", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv1", "inbound_nodes": [[["tf.__operators__.add_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_4_relu1", "inbound_nodes": [[["resblock_part2_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv2", "inbound_nodes": [[["resblock_part2_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"y": ["tf.__operators__.add_6", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv1", "inbound_nodes": [[["tf.__operators__.add_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_5_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_5_relu1", "inbound_nodes": [[["resblock_part2_5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv2", "inbound_nodes": [[["resblock_part2_5_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_5_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"y": ["tf.__operators__.add_7", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv1", "inbound_nodes": [[["tf.__operators__.add_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_6_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_6_relu1", "inbound_nodes": [[["resblock_part2_6_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv2", "inbound_nodes": [[["resblock_part2_6_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_6_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_9", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"y": ["tf.__operators__.add_8", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv1", "inbound_nodes": [[["tf.__operators__.add_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_7_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_7_relu1", "inbound_nodes": [[["resblock_part2_7_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv2", "inbound_nodes": [[["resblock_part2_7_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_10", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_7_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_10", "inbound_nodes": [["tf.math.multiply_10", 0, 0, {"y": ["tf.__operators__.add_9", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv1", "inbound_nodes": [[["tf.__operators__.add_10", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_8_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_8_relu1", "inbound_nodes": [[["resblock_part2_8_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv2", "inbound_nodes": [[["resblock_part2_8_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_11", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_8_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_11", "inbound_nodes": [["tf.math.multiply_11", 0, 0, {"y": ["tf.__operators__.add_10", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_1", "inbound_nodes": [[["tf.__operators__.add_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space", "inbound_nodes": [["upsampler_1", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv1", "inbound_nodes": [[["tf.nn.depth_to_space", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_1_relu1", "inbound_nodes": [[["resblock_part3_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv2", "inbound_nodes": [[["resblock_part3_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_12", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_12", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["tf.math.multiply_12", 0, 0, {"y": ["tf.nn.depth_to_space", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv1", "inbound_nodes": [[["tf.__operators__.add_12", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_2_relu1", "inbound_nodes": [[["resblock_part3_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv2", "inbound_nodes": [[["resblock_part3_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_13", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_13", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.math.multiply_13", 0, 0, {"y": ["tf.__operators__.add_12", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv1", "inbound_nodes": [[["tf.__operators__.add_13", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_3_relu1", "inbound_nodes": [[["resblock_part3_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv2", "inbound_nodes": [[["resblock_part3_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_14", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_14", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["tf.math.multiply_14", 0, 0, {"y": ["tf.__operators__.add_13", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv1", "inbound_nodes": [[["tf.__operators__.add_14", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_4_relu1", "inbound_nodes": [[["resblock_part3_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv2", "inbound_nodes": [[["resblock_part3_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_15", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.math.multiply_15", 0, 0, {"y": ["tf.__operators__.add_14", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "extra_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "extra_conv", "inbound_nodes": [[["tf.__operators__.add_15", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["extra_conv", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_2", "inbound_nodes": [[["tf.__operators__.add_16", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space_1", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space_1", "inbound_nodes": [["upsampler_2", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "output_conv", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_conv", "inbound_nodes": [[["tf.nn.depth_to_space_1", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_conv", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 256, 256]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 256, 256]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "ssi_res_unet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 256, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "input_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "input_conv", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d", "inbound_nodes": [[["input_conv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_1", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv1", "inbound_nodes": [[["downsampler_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_1_relu1", "inbound_nodes": [[["resblock_part1_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_1_conv2", "inbound_nodes": [[["resblock_part1_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["tf.math.multiply", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv1", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_2_relu1", "inbound_nodes": [[["resblock_part1_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_2_conv2", "inbound_nodes": [[["resblock_part1_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"y": ["tf.__operators__.add", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_3_relu1", "inbound_nodes": [[["resblock_part1_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_3_conv2", "inbound_nodes": [[["resblock_part1_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"y": ["tf.__operators__.add_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv1", "inbound_nodes": [[["tf.__operators__.add_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part1_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part1_4_relu1", "inbound_nodes": [[["resblock_part1_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part1_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part1_4_conv2", "inbound_nodes": [[["resblock_part1_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part1_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"y": ["tf.__operators__.add_2", 0, 0], "name": null}]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "name": "zero_padding2d_1", "inbound_nodes": [[["tf.__operators__.add_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "downsampler_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "downsampler_2", "inbound_nodes": [[["zero_padding2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv1", "inbound_nodes": [[["downsampler_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_1_relu1", "inbound_nodes": [[["resblock_part2_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_1_conv2", "inbound_nodes": [[["resblock_part2_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"y": ["downsampler_2", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv1", "inbound_nodes": [[["tf.__operators__.add_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_2_relu1", "inbound_nodes": [[["resblock_part2_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_2_conv2", "inbound_nodes": [[["resblock_part2_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"y": ["tf.__operators__.add_4", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv1", "inbound_nodes": [[["tf.__operators__.add_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_3_relu1", "inbound_nodes": [[["resblock_part2_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_3_conv2", "inbound_nodes": [[["resblock_part2_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"y": ["tf.__operators__.add_5", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv1", "inbound_nodes": [[["tf.__operators__.add_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_4_relu1", "inbound_nodes": [[["resblock_part2_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_4_conv2", "inbound_nodes": [[["resblock_part2_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"y": ["tf.__operators__.add_6", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv1", "inbound_nodes": [[["tf.__operators__.add_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_5_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_5_relu1", "inbound_nodes": [[["resblock_part2_5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_5_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_5_conv2", "inbound_nodes": [[["resblock_part2_5_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_5_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"y": ["tf.__operators__.add_7", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv1", "inbound_nodes": [[["tf.__operators__.add_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_6_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_6_relu1", "inbound_nodes": [[["resblock_part2_6_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_6_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_6_conv2", "inbound_nodes": [[["resblock_part2_6_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_6_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_9", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_9", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"y": ["tf.__operators__.add_8", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv1", "inbound_nodes": [[["tf.__operators__.add_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_7_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_7_relu1", "inbound_nodes": [[["resblock_part2_7_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_7_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_7_conv2", "inbound_nodes": [[["resblock_part2_7_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_10", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_10", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_7_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_10", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_10", "inbound_nodes": [["tf.math.multiply_10", 0, 0, {"y": ["tf.__operators__.add_9", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv1", "inbound_nodes": [[["tf.__operators__.add_10", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part2_8_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part2_8_relu1", "inbound_nodes": [[["resblock_part2_8_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part2_8_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part2_8_conv2", "inbound_nodes": [[["resblock_part2_8_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_11", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_11", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part2_8_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_11", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_11", "inbound_nodes": [["tf.math.multiply_11", 0, 0, {"y": ["tf.__operators__.add_10", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_1", "inbound_nodes": [[["tf.__operators__.add_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space", "inbound_nodes": [["upsampler_1", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv1", "inbound_nodes": [[["tf.nn.depth_to_space", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_1_relu1", "inbound_nodes": [[["resblock_part3_1_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_1_conv2", "inbound_nodes": [[["resblock_part3_1_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_12", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_12", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_1_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["tf.math.multiply_12", 0, 0, {"y": ["tf.nn.depth_to_space", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv1", "inbound_nodes": [[["tf.__operators__.add_12", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_2_relu1", "inbound_nodes": [[["resblock_part3_2_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_2_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_2_conv2", "inbound_nodes": [[["resblock_part3_2_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_13", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_13", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_2_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.math.multiply_13", 0, 0, {"y": ["tf.__operators__.add_12", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv1", "inbound_nodes": [[["tf.__operators__.add_13", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_3_relu1", "inbound_nodes": [[["resblock_part3_3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_3_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_3_conv2", "inbound_nodes": [[["resblock_part3_3_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_14", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_14", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_3_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["tf.math.multiply_14", 0, 0, {"y": ["tf.__operators__.add_13", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv1", "inbound_nodes": [[["tf.__operators__.add_14", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "resblock_part3_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "resblock_part3_4_relu1", "inbound_nodes": [[["resblock_part3_4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "resblock_part3_4_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "resblock_part3_4_conv2", "inbound_nodes": [[["resblock_part3_4_relu1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_15", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["resblock_part3_4_conv2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.math.multiply_15", 0, 0, {"y": ["tf.__operators__.add_14", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "extra_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "extra_conv", "inbound_nodes": [[["tf.__operators__.add_15", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["extra_conv", 0, 0, {"y": ["downsampler_1", 0, 0], "name": null}]]}, {"class_name": "Conv2D", "config": {"name": "upsampler_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "upsampler_2", "inbound_nodes": [[["tf.__operators__.add_16", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.depth_to_space_1", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}, "name": "tf.nn.depth_to_space_1", "inbound_nodes": [["upsampler_2", 0, 0, {"block_size": 2, "data_format": "NCHW"}]]}, {"class_name": "Conv2D", "config": {"name": "output_conv", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_conv", "inbound_nodes": [[["tf.nn.depth_to_space_1", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_conv", 0, 0]]}}}
"
_tf_keras_input_layerä{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 256, 256]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 256, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
ý	

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Conv2D", "name": "input_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "input_conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 28}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 256, 256]}}

i	variables
jtrainable_variables
kregularization_losses
l	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"÷
_tf_keras_layerÝ{"class_name": "ZeroPadding2D", "name": "zero_padding2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}



mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
á__call__
+â&call_and_return_all_conditional_losses"Ý
_tf_keras_layerÃ{"class_name": "Conv2D", "name": "downsampler_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "downsampler_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 258, 258]}}



skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

y	variables
ztrainable_variables
{regularization_losses
|	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_1_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}



}kernel
~bias
	variables
trainable_variables
regularization_losses
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
	variables
trainable_variables
regularization_losses
	keras_api
é__call__
+ê&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

	variables
trainable_variables
regularization_losses
	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_2_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
	variables
trainable_variables
regularization_losses
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
	variables
trainable_variables
regularization_losses
	keras_api
ï__call__
+ð&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

	variables
trainable_variables
regularization_losses
 	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_3_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


¡kernel
	¢bias
£	variables
¤trainable_variables
¥regularization_losses
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
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part1_4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

¯	variables
°trainable_variables
±regularization_losses
²	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part1_4_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part1_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


³kernel
	´bias
µ	variables
¶trainable_variables
·regularization_losses
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
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
û__call__
+ü&call_and_return_all_conditional_losses"û
_tf_keras_layerá{"class_name": "ZeroPadding2D", "name": "zero_padding2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}


¿kernel
	Àbias
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"Ý
_tf_keras_layerÃ{"class_name": "Conv2D", "name": "downsampler_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "downsampler_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 130, 130]}}


Åkernel
	Æbias
Ç	variables
Ètrainable_variables
Éregularization_losses
Ê	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_1_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


Ïkernel
	Ðbias
Ñ	variables
Òtrainable_variables
Óregularization_losses
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
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_2_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


ákernel
	âbias
ã	variables
ätrainable_variables
åregularization_losses
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
ë	variables
ìtrainable_variables
íregularization_losses
î	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_3_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


ókernel
	ôbias
õ	variables
ötrainable_variables
÷regularization_losses
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
ý	variables
þtrainable_variables
ÿregularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_4_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
	variables
trainable_variables
regularization_losses
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_5_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_5_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_5_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_5_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
	variables
trainable_variables
regularization_losses
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
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_6_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_6_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
__call__
+ &call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_6_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_6_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


©kernel
	ªbias
«	variables
¬trainable_variables
­regularization_losses
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
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_7_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_7_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_7_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_7_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


»kernel
	¼bias
½	variables
¾trainable_variables
¿regularization_losses
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
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Conv2D", "name": "resblock_part2_8_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_8_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}

É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part2_8_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part2_8_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


Íkernel
	Îbias
Ï	variables
Ðtrainable_variables
Ñregularization_losses
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
×	variables
Øtrainable_variables
Ùregularization_losses
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
Þ	variables
ßtrainable_variables
àregularization_losses
á	keras_api
±__call__
+²&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

â	variables
ãtrainable_variables
äregularization_losses
å	keras_api
³__call__
+´&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_1_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_1_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


ækernel
	çbias
è	variables
étrainable_variables
êregularization_losses
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
ð	variables
ñtrainable_variables
òregularization_losses
ó	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_2_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

ô	variables
õtrainable_variables
öregularization_losses
÷	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_2_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_2_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


økernel
	ùbias
ú	variables
ûtrainable_variables
üregularization_losses
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
	variables
trainable_variables
regularization_losses
	keras_api
½__call__
+¾&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_3_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

	variables
trainable_variables
regularization_losses
	keras_api
¿__call__
+À&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_3_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_3_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
	variables
trainable_variables
regularization_losses
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
	variables
trainable_variables
regularization_losses
	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Conv2D", "name": "resblock_part3_4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_4_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}

	variables
trainable_variables
regularization_losses
	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "ReLU", "name": "resblock_part3_4_relu1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "resblock_part3_4_relu1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}


kernel
	bias
	variables
trainable_variables
 regularization_losses
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
¦	variables
§trainable_variables
¨regularization_losses
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
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"Ù
_tf_keras_layer¿{"class_name": "Conv2D", "name": "upsampler_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsampler_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 128, 128]}}
÷
±	keras_api"ä
_tf_keras_layerÊ{"class_name": "TFOpLambda", "name": "tf.nn.depth_to_space_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.depth_to_space_1", "trainable": true, "dtype": "float32", "function": "nn.depth_to_space"}}


²kernel
	³bias
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Conv2D", "name": "output_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_conv", "trainable": true, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 256, 256]}}
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
 "
trackable_list_wrapper
Ó
¸layers
¹layer_metrics
^	variables
ºnon_trainable_variables
»metrics
_trainable_variables
 ¼layer_regularization_losses
`regularization_losses
Û__call__
Ú_default_save_signature
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
-
Ïserving_default"
signature_map
+:)@2input_conv/kernel
:@2input_conv/bias
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
½layers
¾layer_metrics
e	variables
¿non_trainable_variables
Àmetrics
ftrainable_variables
 Álayer_regularization_losses
gregularization_losses
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
Âlayers
Ãlayer_metrics
i	variables
Änon_trainable_variables
Åmetrics
jtrainable_variables
 Ælayer_regularization_losses
kregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
.:,@@2downsampler_1/kernel
 :@2downsampler_1/bias
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Çlayers
Èlayer_metrics
o	variables
Énon_trainable_variables
Êmetrics
ptrainable_variables
 Ëlayer_regularization_losses
qregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_1_conv1/kernel
):'@2resblock_part1_1_conv1/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ìlayers
Ílayer_metrics
u	variables
Înon_trainable_variables
Ïmetrics
vtrainable_variables
 Ðlayer_regularization_losses
wregularization_losses
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
Ñlayers
Òlayer_metrics
y	variables
Ónon_trainable_variables
Ômetrics
ztrainable_variables
 Õlayer_regularization_losses
{regularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_1_conv2/kernel
):'@2resblock_part1_1_conv2/bias
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
·
Ölayers
×layer_metrics
	variables
Ønon_trainable_variables
Ùmetrics
trainable_variables
 Úlayer_regularization_losses
regularization_losses
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
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûlayers
Ülayer_metrics
	variables
Ýnon_trainable_variables
Þmetrics
trainable_variables
 ßlayer_regularization_losses
regularization_losses
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
àlayers
álayer_metrics
	variables
ânon_trainable_variables
ãmetrics
trainable_variables
 älayer_regularization_losses
regularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_2_conv2/kernel
):'@2resblock_part1_2_conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ålayers
ælayer_metrics
	variables
çnon_trainable_variables
èmetrics
trainable_variables
 élayer_regularization_losses
regularization_losses
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
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
êlayers
ëlayer_metrics
	variables
ìnon_trainable_variables
ímetrics
trainable_variables
 îlayer_regularization_losses
regularization_losses
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
ïlayers
ðlayer_metrics
	variables
ñnon_trainable_variables
òmetrics
trainable_variables
 ólayer_regularization_losses
regularization_losses
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_3_conv2/kernel
):'@2resblock_part1_3_conv2/bias
0
¡0
¢1"
trackable_list_wrapper
0
¡0
¢1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ôlayers
õlayer_metrics
£	variables
önon_trainable_variables
÷metrics
¤trainable_variables
 ølayer_regularization_losses
¥regularization_losses
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
0
©0
ª1"
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ùlayers
úlayer_metrics
«	variables
ûnon_trainable_variables
ümetrics
¬trainable_variables
 ýlayer_regularization_losses
­regularization_losses
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
þlayers
ÿlayer_metrics
¯	variables
non_trainable_variables
metrics
°trainable_variables
 layer_regularization_losses
±regularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part1_4_conv2/kernel
):'@2resblock_part1_4_conv2/bias
0
³0
´1"
trackable_list_wrapper
0
³0
´1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
µ	variables
non_trainable_variables
metrics
¶trainable_variables
 layer_regularization_losses
·regularization_losses
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
layers
layer_metrics
»	variables
non_trainable_variables
metrics
¼trainable_variables
 layer_regularization_losses
½regularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
.:,@@2downsampler_2/kernel
 :@2downsampler_2/bias
0
¿0
À1"
trackable_list_wrapper
0
¿0
À1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
Á	variables
non_trainable_variables
metrics
Âtrainable_variables
 layer_regularization_losses
Ãregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_1_conv1/kernel
):'@2resblock_part2_1_conv1/bias
0
Å0
Æ1"
trackable_list_wrapper
0
Å0
Æ1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
Ç	variables
non_trainable_variables
metrics
Ètrainable_variables
 layer_regularization_losses
Éregularization_losses
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
layers
layer_metrics
Ë	variables
non_trainable_variables
metrics
Ìtrainable_variables
 layer_regularization_losses
Íregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_1_conv2/kernel
):'@2resblock_part2_1_conv2/bias
0
Ï0
Ð1"
trackable_list_wrapper
0
Ï0
Ð1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
Ñ	variables
non_trainable_variables
metrics
Òtrainable_variables
  layer_regularization_losses
Óregularization_losses
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
0
×0
Ø1"
trackable_list_wrapper
0
×0
Ø1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¡layers
¢layer_metrics
Ù	variables
£non_trainable_variables
¤metrics
Útrainable_variables
 ¥layer_regularization_losses
Ûregularization_losses
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
¦layers
§layer_metrics
Ý	variables
¨non_trainable_variables
©metrics
Þtrainable_variables
 ªlayer_regularization_losses
ßregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_2_conv2/kernel
):'@2resblock_part2_2_conv2/bias
0
á0
â1"
trackable_list_wrapper
0
á0
â1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
«layers
¬layer_metrics
ã	variables
­non_trainable_variables
®metrics
ätrainable_variables
 ¯layer_regularization_losses
åregularization_losses
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
0
é0
ê1"
trackable_list_wrapper
0
é0
ê1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
°layers
±layer_metrics
ë	variables
²non_trainable_variables
³metrics
ìtrainable_variables
 ´layer_regularization_losses
íregularization_losses
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
µlayers
¶layer_metrics
ï	variables
·non_trainable_variables
¸metrics
ðtrainable_variables
 ¹layer_regularization_losses
ñregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_3_conv2/kernel
):'@2resblock_part2_3_conv2/bias
0
ó0
ô1"
trackable_list_wrapper
0
ó0
ô1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ºlayers
»layer_metrics
õ	variables
¼non_trainable_variables
½metrics
ötrainable_variables
 ¾layer_regularization_losses
÷regularization_losses
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
0
û0
ü1"
trackable_list_wrapper
0
û0
ü1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¿layers
Àlayer_metrics
ý	variables
Ánon_trainable_variables
Âmetrics
þtrainable_variables
 Ãlayer_regularization_losses
ÿregularization_losses
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
Älayers
Ålayer_metrics
	variables
Ænon_trainable_variables
Çmetrics
trainable_variables
 Èlayer_regularization_losses
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_4_conv2/kernel
):'@2resblock_part2_4_conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Élayers
Êlayer_metrics
	variables
Ënon_trainable_variables
Ìmetrics
trainable_variables
 Ílayer_regularization_losses
regularization_losses
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
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Îlayers
Ïlayer_metrics
	variables
Ðnon_trainable_variables
Ñmetrics
trainable_variables
 Òlayer_regularization_losses
regularization_losses
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
Ólayers
Ôlayer_metrics
	variables
Õnon_trainable_variables
Ömetrics
trainable_variables
 ×layer_regularization_losses
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_5_conv2/kernel
):'@2resblock_part2_5_conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ølayers
Ùlayer_metrics
	variables
Únon_trainable_variables
Ûmetrics
trainable_variables
 Ülayer_regularization_losses
regularization_losses
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
0
0
 1"
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ýlayers
Þlayer_metrics
¡	variables
ßnon_trainable_variables
àmetrics
¢trainable_variables
 álayer_regularization_losses
£regularization_losses
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
âlayers
ãlayer_metrics
¥	variables
änon_trainable_variables
åmetrics
¦trainable_variables
 ælayer_regularization_losses
§regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_6_conv2/kernel
):'@2resblock_part2_6_conv2/bias
0
©0
ª1"
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
çlayers
èlayer_metrics
«	variables
énon_trainable_variables
êmetrics
¬trainable_variables
 ëlayer_regularization_losses
­regularization_losses
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
0
±0
²1"
trackable_list_wrapper
0
±0
²1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ìlayers
ílayer_metrics
³	variables
înon_trainable_variables
ïmetrics
´trainable_variables
 ðlayer_regularization_losses
µregularization_losses
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
ñlayers
òlayer_metrics
·	variables
ónon_trainable_variables
ômetrics
¸trainable_variables
 õlayer_regularization_losses
¹regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_7_conv2/kernel
):'@2resblock_part2_7_conv2/bias
0
»0
¼1"
trackable_list_wrapper
0
»0
¼1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ölayers
÷layer_metrics
½	variables
ønon_trainable_variables
ùmetrics
¾trainable_variables
 úlayer_regularization_losses
¿regularization_losses
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
0
Ã0
Ä1"
trackable_list_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ûlayers
ülayer_metrics
Å	variables
ýnon_trainable_variables
þmetrics
Ætrainable_variables
 ÿlayer_regularization_losses
Çregularization_losses
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
layers
layer_metrics
É	variables
non_trainable_variables
metrics
Êtrainable_variables
 layer_regularization_losses
Ëregularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part2_8_conv2/kernel
):'@2resblock_part2_8_conv2/bias
0
Í0
Î1"
trackable_list_wrapper
0
Í0
Î1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
Ï	variables
non_trainable_variables
metrics
Ðtrainable_variables
 layer_regularization_losses
Ñregularization_losses
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
0
Õ0
Ö1"
trackable_list_wrapper
0
Õ0
Ö1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
×	variables
non_trainable_variables
metrics
Øtrainable_variables
 layer_regularization_losses
Ùregularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
7:5@@2resblock_part3_1_conv1/kernel
):'@2resblock_part3_1_conv1/bias
0
Ü0
Ý1"
trackable_list_wrapper
0
Ü0
Ý1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
Þ	variables
non_trainable_variables
metrics
ßtrainable_variables
 layer_regularization_losses
àregularization_losses
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
layers
layer_metrics
â	variables
non_trainable_variables
metrics
ãtrainable_variables
 layer_regularization_losses
äregularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_1_conv2/kernel
):'@2resblock_part3_1_conv2/bias
0
æ0
ç1"
trackable_list_wrapper
0
æ0
ç1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
è	variables
non_trainable_variables
metrics
étrainable_variables
 layer_regularization_losses
êregularization_losses
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
0
î0
ï1"
trackable_list_wrapper
0
î0
ï1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
ð	variables
 non_trainable_variables
¡metrics
ñtrainable_variables
 ¢layer_regularization_losses
òregularization_losses
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
£layers
¤layer_metrics
ô	variables
¥non_trainable_variables
¦metrics
õtrainable_variables
 §layer_regularization_losses
öregularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_2_conv2/kernel
):'@2resblock_part3_2_conv2/bias
0
ø0
ù1"
trackable_list_wrapper
0
ø0
ù1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¨layers
©layer_metrics
ú	variables
ªnon_trainable_variables
«metrics
ûtrainable_variables
 ¬layer_regularization_losses
üregularization_losses
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
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
­layers
®layer_metrics
	variables
¯non_trainable_variables
°metrics
trainable_variables
 ±layer_regularization_losses
regularization_losses
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
²layers
³layer_metrics
	variables
´non_trainable_variables
µmetrics
trainable_variables
 ¶layer_regularization_losses
regularization_losses
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_3_conv2/kernel
):'@2resblock_part3_3_conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
·layers
¸layer_metrics
	variables
¹non_trainable_variables
ºmetrics
trainable_variables
 »layer_regularization_losses
regularization_losses
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
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¼layers
½layer_metrics
	variables
¾non_trainable_variables
¿metrics
trainable_variables
 Àlayer_regularization_losses
regularization_losses
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
Álayers
Âlayer_metrics
	variables
Ãnon_trainable_variables
Ämetrics
trainable_variables
 Ålayer_regularization_losses
regularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
7:5@@2resblock_part3_4_conv2/kernel
):'@2resblock_part3_4_conv2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ælayers
Çlayer_metrics
	variables
Ènon_trainable_variables
Émetrics
trainable_variables
 Êlayer_regularization_losses
 regularization_losses
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
0
¤0
¥1"
trackable_list_wrapper
0
¤0
¥1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ëlayers
Ìlayer_metrics
¦	variables
Ínon_trainable_variables
Îmetrics
§trainable_variables
 Ïlayer_regularization_losses
¨regularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
-:+@2upsampler_2/kernel
:2upsampler_2/bias
0
«0
¬1"
trackable_list_wrapper
0
«0
¬1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ðlayers
Ñlayer_metrics
­	variables
Ònon_trainable_variables
Ómetrics
®trainable_variables
 Ôlayer_regularization_losses
¯regularization_losses
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
,:*@2output_conv/kernel
:2output_conv/bias
0
²0
³1"
trackable_list_wrapper
0
²0
³1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Õlayers
Ölayer_metrics
´	variables
×non_trainable_variables
Ømetrics
µtrainable_variables
 Ùlayer_regularization_losses
¶regularization_losses
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ë2è
__inference__wrapped_model_2954Ä
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
input_layerÿÿÿÿÿÿÿÿÿ
ú2÷
+__inference_ssi_res_unet_layer_call_fn_5451
+__inference_ssi_res_unet_layer_call_fn_4990
+__inference_ssi_res_unet_layer_call_fn_6457
+__inference_ssi_res_unet_layer_call_fn_6650À
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
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5955
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4528
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_6264
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4260À
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
Ó2Ð
)__inference_input_conv_layer_call_fn_6669¢
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
D__inference_input_conv_layer_call_and_return_conditional_losses_6660¢
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
-__inference_zero_padding2d_layer_call_fn_2967à
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
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_2961à
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
,__inference_downsampler_1_layer_call_fn_6688¢
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
G__inference_downsampler_1_layer_call_and_return_conditional_losses_6679¢
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
5__inference_resblock_part1_1_conv1_layer_call_fn_6707¢
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
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_6698¢
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
5__inference_resblock_part1_1_relu1_layer_call_fn_6717¢
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
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_6712¢
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
5__inference_resblock_part1_1_conv2_layer_call_fn_6736¢
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
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_6727¢
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
5__inference_resblock_part1_2_conv1_layer_call_fn_6755¢
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
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_6746¢
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
5__inference_resblock_part1_2_relu1_layer_call_fn_6765¢
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
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_6760¢
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
5__inference_resblock_part1_2_conv2_layer_call_fn_6784¢
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
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_6775¢
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
5__inference_resblock_part1_3_conv1_layer_call_fn_6803¢
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
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_6794¢
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
5__inference_resblock_part1_3_relu1_layer_call_fn_6813¢
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
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_6808¢
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
5__inference_resblock_part1_3_conv2_layer_call_fn_6832¢
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
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_6823¢
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
5__inference_resblock_part1_4_conv1_layer_call_fn_6851¢
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
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_6842¢
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
5__inference_resblock_part1_4_relu1_layer_call_fn_6861¢
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
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_6856¢
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
5__inference_resblock_part1_4_conv2_layer_call_fn_6880¢
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
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_6871¢
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
/__inference_zero_padding2d_1_layer_call_fn_2980à
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
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2974à
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
,__inference_downsampler_2_layer_call_fn_6899¢
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
G__inference_downsampler_2_layer_call_and_return_conditional_losses_6890¢
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
5__inference_resblock_part2_1_conv1_layer_call_fn_6918¢
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
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_6909¢
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
5__inference_resblock_part2_1_relu1_layer_call_fn_6928¢
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
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_6923¢
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
5__inference_resblock_part2_1_conv2_layer_call_fn_6947¢
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
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_6938¢
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
5__inference_resblock_part2_2_conv1_layer_call_fn_6966¢
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
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_6957¢
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
5__inference_resblock_part2_2_relu1_layer_call_fn_6976¢
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
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_6971¢
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
5__inference_resblock_part2_2_conv2_layer_call_fn_6995¢
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
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_6986¢
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
5__inference_resblock_part2_3_conv1_layer_call_fn_7014¢
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
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_7005¢
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
5__inference_resblock_part2_3_relu1_layer_call_fn_7024¢
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
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_7019¢
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
5__inference_resblock_part2_3_conv2_layer_call_fn_7043¢
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
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_7034¢
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
5__inference_resblock_part2_4_conv1_layer_call_fn_7062¢
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
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_7053¢
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
5__inference_resblock_part2_4_relu1_layer_call_fn_7072¢
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
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_7067¢
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
5__inference_resblock_part2_4_conv2_layer_call_fn_7091¢
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
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_7082¢
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
5__inference_resblock_part2_5_conv1_layer_call_fn_7110¢
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
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_7101¢
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
5__inference_resblock_part2_5_relu1_layer_call_fn_7120¢
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
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_7115¢
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
5__inference_resblock_part2_5_conv2_layer_call_fn_7139¢
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
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_7130¢
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
5__inference_resblock_part2_6_conv1_layer_call_fn_7158¢
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
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_7149¢
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
5__inference_resblock_part2_6_relu1_layer_call_fn_7168¢
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
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_7163¢
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
5__inference_resblock_part2_6_conv2_layer_call_fn_7187¢
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
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_7178¢
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
5__inference_resblock_part2_7_conv1_layer_call_fn_7206¢
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
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_7197¢
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
5__inference_resblock_part2_7_relu1_layer_call_fn_7216¢
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
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_7211¢
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
5__inference_resblock_part2_7_conv2_layer_call_fn_7235¢
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
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_7226¢
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
5__inference_resblock_part2_8_conv1_layer_call_fn_7254¢
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
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_7245¢
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
5__inference_resblock_part2_8_relu1_layer_call_fn_7264¢
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
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_7259¢
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
5__inference_resblock_part2_8_conv2_layer_call_fn_7283¢
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
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_7274¢
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
*__inference_upsampler_1_layer_call_fn_7302¢
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
E__inference_upsampler_1_layer_call_and_return_conditional_losses_7293¢
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
5__inference_resblock_part3_1_conv1_layer_call_fn_7321¢
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
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_7312¢
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
5__inference_resblock_part3_1_relu1_layer_call_fn_7331¢
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
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_7326¢
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
5__inference_resblock_part3_1_conv2_layer_call_fn_7350¢
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
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_7341¢
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
5__inference_resblock_part3_2_conv1_layer_call_fn_7369¢
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
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_7360¢
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
5__inference_resblock_part3_2_relu1_layer_call_fn_7379¢
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
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_7374¢
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
5__inference_resblock_part3_2_conv2_layer_call_fn_7398¢
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
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_7389¢
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
5__inference_resblock_part3_3_conv1_layer_call_fn_7417¢
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
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_7408¢
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
5__inference_resblock_part3_3_relu1_layer_call_fn_7427¢
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
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_7422¢
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
5__inference_resblock_part3_3_conv2_layer_call_fn_7446¢
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
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_7437¢
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
5__inference_resblock_part3_4_conv1_layer_call_fn_7465¢
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
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_7456¢
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
5__inference_resblock_part3_4_relu1_layer_call_fn_7475¢
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
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_7470¢
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
5__inference_resblock_part3_4_conv2_layer_call_fn_7494¢
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
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_7485¢
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
)__inference_extra_conv_layer_call_fn_7513¢
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
D__inference_extra_conv_layer_call_and_return_conditional_losses_7504¢
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
*__inference_upsampler_2_layer_call_fn_7532¢
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
E__inference_upsampler_2_layer_call_and_return_conditional_losses_7523¢
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
*__inference_output_conv_layer_call_fn_7551¢
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
E__inference_output_conv_layer_call_and_return_conditional_losses_7542¢
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
"__inference_signature_wrapper_5646input_layer"
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
__inference__wrapped_model_2954¼´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³>¢;
4¢1
/,
input_layerÿÿÿÿÿÿÿÿÿ
ª "Cª@
>
output_conv/,
output_convÿÿÿÿÿÿÿÿÿ»
G__inference_downsampler_1_layer_call_and_return_conditional_losses_6679pmn9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_downsampler_1_layer_call_fn_6688cmn9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@»
G__inference_downsampler_2_layer_call_and_return_conditional_losses_6890p¿À9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
,__inference_downsampler_2_layer_call_fn_6899c¿À9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@@@º
D__inference_extra_conv_layer_call_and_return_conditional_losses_7504r¤¥9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_extra_conv_layer_call_fn_7513e¤¥9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@¸
D__inference_input_conv_layer_call_and_return_conditional_losses_6660pcd9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_input_conv_layer_call_fn_6669ccd9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ@»
E__inference_output_conv_layer_call_and_return_conditional_losses_7542r²³9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_output_conv_layer_call_fn_7551e²³9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿÄ
P__inference_resblock_part1_1_conv1_layer_call_and_return_conditional_losses_6698pst9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_1_conv1_layer_call_fn_6707cst9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Ä
P__inference_resblock_part1_1_conv2_layer_call_and_return_conditional_losses_6727p}~9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_1_conv2_layer_call_fn_6736c}~9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_1_relu1_layer_call_and_return_conditional_losses_6712l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_1_relu1_layer_call_fn_6717_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_2_conv1_layer_call_and_return_conditional_losses_6746r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_2_conv1_layer_call_fn_6755e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_2_conv2_layer_call_and_return_conditional_losses_6775r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_2_conv2_layer_call_fn_6784e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_2_relu1_layer_call_and_return_conditional_losses_6760l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_2_relu1_layer_call_fn_6765_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_3_conv1_layer_call_and_return_conditional_losses_6794r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_3_conv1_layer_call_fn_6803e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_3_conv2_layer_call_and_return_conditional_losses_6823r¡¢9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_3_conv2_layer_call_fn_6832e¡¢9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_3_relu1_layer_call_and_return_conditional_losses_6808l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_3_relu1_layer_call_fn_6813_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_4_conv1_layer_call_and_return_conditional_losses_6842r©ª9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_4_conv1_layer_call_fn_6851e©ª9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part1_4_conv2_layer_call_and_return_conditional_losses_6871r³´9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_4_conv2_layer_call_fn_6880e³´9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part1_4_relu1_layer_call_and_return_conditional_losses_6856l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part1_4_relu1_layer_call_fn_6861_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Â
P__inference_resblock_part2_1_conv1_layer_call_and_return_conditional_losses_6909nÅÆ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_1_conv1_layer_call_fn_6918aÅÆ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_1_conv2_layer_call_and_return_conditional_losses_6938nÏÐ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_1_conv2_layer_call_fn_6947aÏÐ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_1_relu1_layer_call_and_return_conditional_losses_6923h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_1_relu1_layer_call_fn_6928[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_2_conv1_layer_call_and_return_conditional_losses_6957n×Ø7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_2_conv1_layer_call_fn_6966a×Ø7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_2_conv2_layer_call_and_return_conditional_losses_6986náâ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_2_conv2_layer_call_fn_6995aáâ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_2_relu1_layer_call_and_return_conditional_losses_6971h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_2_relu1_layer_call_fn_6976[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_3_conv1_layer_call_and_return_conditional_losses_7005néê7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_3_conv1_layer_call_fn_7014aéê7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_3_conv2_layer_call_and_return_conditional_losses_7034nóô7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_3_conv2_layer_call_fn_7043aóô7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_3_relu1_layer_call_and_return_conditional_losses_7019h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_3_relu1_layer_call_fn_7024[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_4_conv1_layer_call_and_return_conditional_losses_7053nûü7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_4_conv1_layer_call_fn_7062aûü7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_4_conv2_layer_call_and_return_conditional_losses_7082n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_4_conv2_layer_call_fn_7091a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_4_relu1_layer_call_and_return_conditional_losses_7067h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_4_relu1_layer_call_fn_7072[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_5_conv1_layer_call_and_return_conditional_losses_7101n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_5_conv1_layer_call_fn_7110a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_5_conv2_layer_call_and_return_conditional_losses_7130n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_5_conv2_layer_call_fn_7139a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_5_relu1_layer_call_and_return_conditional_losses_7115h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_5_relu1_layer_call_fn_7120[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_6_conv1_layer_call_and_return_conditional_losses_7149n 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_6_conv1_layer_call_fn_7158a 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_6_conv2_layer_call_and_return_conditional_losses_7178n©ª7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_6_conv2_layer_call_fn_7187a©ª7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_6_relu1_layer_call_and_return_conditional_losses_7163h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_6_relu1_layer_call_fn_7168[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_7_conv1_layer_call_and_return_conditional_losses_7197n±²7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_7_conv1_layer_call_fn_7206a±²7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_7_conv2_layer_call_and_return_conditional_losses_7226n»¼7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_7_conv2_layer_call_fn_7235a»¼7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_7_relu1_layer_call_and_return_conditional_losses_7211h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_7_relu1_layer_call_fn_7216[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_8_conv1_layer_call_and_return_conditional_losses_7245nÃÄ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_8_conv1_layer_call_fn_7254aÃÄ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Â
P__inference_resblock_part2_8_conv2_layer_call_and_return_conditional_losses_7274nÍÎ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_8_conv2_layer_call_fn_7283aÍÎ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@¼
P__inference_resblock_part2_8_relu1_layer_call_and_return_conditional_losses_7259h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
5__inference_resblock_part2_8_relu1_layer_call_fn_7264[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@Æ
P__inference_resblock_part3_1_conv1_layer_call_and_return_conditional_losses_7312rÜÝ9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_1_conv1_layer_call_fn_7321eÜÝ9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_1_conv2_layer_call_and_return_conditional_losses_7341ræç9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_1_conv2_layer_call_fn_7350eæç9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_1_relu1_layer_call_and_return_conditional_losses_7326l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_1_relu1_layer_call_fn_7331_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_2_conv1_layer_call_and_return_conditional_losses_7360rîï9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_2_conv1_layer_call_fn_7369eîï9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_2_conv2_layer_call_and_return_conditional_losses_7389røù9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_2_conv2_layer_call_fn_7398eøù9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_2_relu1_layer_call_and_return_conditional_losses_7374l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_2_relu1_layer_call_fn_7379_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_3_conv1_layer_call_and_return_conditional_losses_7408r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_3_conv1_layer_call_fn_7417e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_3_conv2_layer_call_and_return_conditional_losses_7437r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_3_conv2_layer_call_fn_7446e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_3_relu1_layer_call_and_return_conditional_losses_7422l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_3_relu1_layer_call_fn_7427_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_4_conv1_layer_call_and_return_conditional_losses_7456r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_4_conv1_layer_call_fn_7465e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@Æ
P__inference_resblock_part3_4_conv2_layer_call_and_return_conditional_losses_7485r9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_4_conv2_layer_call_fn_7494e9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@À
P__inference_resblock_part3_4_relu1_layer_call_and_return_conditional_losses_7470l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_resblock_part3_4_relu1_layer_call_fn_7475_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@ò
"__inference_signature_wrapper_5646Ë´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³M¢J
¢ 
Cª@
>
input_layer/,
input_layerÿÿÿÿÿÿÿÿÿ"Cª@
>
output_conv/,
output_convÿÿÿÿÿÿÿÿÿû
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4260°´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 û
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_4528°´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ö
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_5955«´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ö
F__inference_ssi_res_unet_layer_call_and_return_conditional_losses_6264«´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ó
+__inference_ssi_res_unet_layer_call_fn_4990£´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿÓ
+__inference_ssi_res_unet_layer_call_fn_5451£´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿÎ
+__inference_ssi_res_unet_layer_call_fn_6457´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿÎ
+__inference_ssi_res_unet_layer_call_fn_6650´cdmnst}~ÐÑ¡¢Ò©ª³´Ó¿ÀÅÆÏÐÔ×ØáâÕéêóôÖûü×Ø ©ªÙ±²»¼ÚÃÄÍÎÛÕÖÜÝæçÜîïøùÝÞß¤¥«¬²³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ¸
E__inference_upsampler_1_layer_call_and_return_conditional_losses_7293oÕÖ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
*__inference_upsampler_1_layer_call_fn_7302bÕÖ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "!ÿÿÿÿÿÿÿÿÿ@@¼
E__inference_upsampler_2_layer_call_and_return_conditional_losses_7523s«¬9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "0¢-
&#
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_upsampler_2_layer_call_fn_7532f«¬9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "# ÿÿÿÿÿÿÿÿÿí
J__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2974R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_zero_padding2d_1_layer_call_fn_2980R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_2961R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_zero_padding2d_layer_call_fn_2967R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ