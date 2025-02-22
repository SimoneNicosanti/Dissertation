\* Partitioning *\
Minimize
OBJ: 848692.2 x_(Conv__435)(server_0) + 84869.17 x_(Conv__435)(server_1)
 + 5734.40675676 x_(resnet50_conv1_relu_Relu)(server_0)
 + 573.440337838 x_(resnet50_conv1_relu_Relu)(server_1)
 + 93184.1097973 x_(resnet50_conv2_block1_1_conv_Conv2D)(server_0)
 + 9318.40548986 x_(resnet50_conv2_block1_1_conv_Conv2D)(server_1)
 + 1433.60168919 x_(resnet50_conv2_block1_1_relu_Relu)(server_0)
 + 143.360084459 x_(resnet50_conv2_block1_1_relu_Relu)(server_1)
 + 827188.174662 x_(resnet50_conv2_block1_2_conv_Conv2D)(server_0)
 + 82718.7687331 x_(resnet50_conv2_block1_2_conv_Conv2D)(server_1)
 + 1433.60168919 x_(resnet50_conv2_block1_2_relu_Relu)(server_0)
 + 143.360084459 x_(resnet50_conv2_block1_2_relu_Relu)(server_1)
 + 372736.439189 x_(resnet50_conv2_block1_3_conv_Conv2D)(server_0)
 + 37273.6219595 x_(resnet50_conv2_block1_3_conv_Conv2D)(server_1)
 + 12902.4152027 x_(resnet50_pool1_pool_MaxPool)(server_0)
 + 1290.24076014 x_(resnet50_pool1_pool_MaxPool)(server_1)
 + 1606.59722007 y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_0)_(server_1))
 + 3212.22922007 y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_1)_(server_0))
 + 301.236978763 y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_0)_(server_1))
 + 602.292978763 y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_1)_(server_0))
 + 301.236978763 y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_0)_(server_1))
 + 602.292978763 y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_1)_(server_0))
 + 1606.59722007 y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_0)_(server_1))
 + 3212.22922007 y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_1)_(server_0))
 + 401.649305017 y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_0)_(server_1))
 + 803.057305017 y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_1)_(server_0))
 + 401.649305017 y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_0)_(server_1))
 + 803.057305017 y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_1)_(server_0))
 + 401.649305017 y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_0)_(server_1))
 + 803.057305017 y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_1)_(server_0))
 + 401.649305017 y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_0)_(server_1))
 + 803.057305017 y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_1)_(server_0))
 + 1664.488 y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_0)_(server_1))
 + 3327.976 y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_1)_(server_0))
 + 401.649305017 y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_0)_(server_1))
 + 803.057305017 y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_1)_(server_0))

Subject To
_C1: x_(args_0_wrapper)(server_0) + x_(args_0_wrapper)(server_1) = 1
_C10: x_(resnet50_conv2_block1_2_relu_Relu)(server_0)
 + x_(resnet50_conv2_block1_2_relu_Relu)(server_1) = 1
_C11: x_(resnet50_conv2_block1_3_conv_Conv2D)(server_0)
 + x_(resnet50_conv2_block1_3_conv_Conv2D)(server_1) = 1
_C12:
 y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_0)_(server_0))
 + y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_0)_(server_1))
 + y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_1)_(server_0))
 + y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_1)_(server_1))
 = 1
_C13: y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_0)_(server_0))
 + y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_0)_(server_1))
 + y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_1)_(server_0))
 + y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_1)_(server_1)) = 1
_C14: y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_0)_(server_0))
 + y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_0)_(server_1))
 + y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_1)_(server_0))
 + y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_1)_(server_1)) = 1
_C15:
 y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_0)_(server_0))
 + y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_0)_(server_1))
 + y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_1)_(server_0))
 + y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_1)_(server_1))
 = 1
_C16:
 y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_0)_(server_0))
 + y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_0)_(server_1))
 + y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_1)_(server_0))
 + y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_1)_(server_1))
 = 1
_C17:
 y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_0)_(server_0))
 + y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_0)_(server_1))
 + y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_1)_(server_0))
 + y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_1)_(server_1))
 = 1
_C18:
 y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_0)_(server_0))
 + y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_0)_(server_1))
 + y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_1)_(server_0))
 + y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_1)_(server_1))
 = 1
_C19:
 y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_0)_(server_0))
 + y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_0)_(server_1))
 + y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_1)_(server_0))
 + y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_1)_(server_1))
 = 1
_C2: x_(resnet50_conv1_conv_Conv2D__6)(server_0)
 + x_(resnet50_conv1_conv_Conv2D__6)(server_1) = 1
_C20:
 y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_0)_(server_0))
 + y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_0)_(server_1))
 + y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_1)_(server_0))
 + y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_1)_(server_1))
 = 1
_C21:
 y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_0)_(server_0))
 + y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_0)_(server_1))
 + y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_1)_(server_0))
 + y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_1)_(server_1))
 = 1
_C22: x_(args_0_wrapper)(server_0)
 - y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_0)_(server_0))
 - y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_0)_(server_1))
 = 0
_C23: x_(args_0_wrapper)(server_1)
 - y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_1)_(server_0))
 - y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_1)_(server_1))
 = 0
_C24: x_(resnet50_conv1_conv_Conv2D__6)(server_0)
 - y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_0)_(server_0))
 - y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_0)_(server_1)) = 0
_C25: x_(resnet50_conv1_conv_Conv2D__6)(server_1)
 - y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_1)_(server_0))
 - y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_1)_(server_1)) = 0
_C26: x_(Conv__435)(server_0)
 - y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_0)_(server_0))
 - y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_0)_(server_1)) = 0
_C27: x_(Conv__435)(server_1)
 - y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_1)_(server_0))
 - y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_1)_(server_1)) = 0
_C28: x_(resnet50_conv1_relu_Relu)(server_0)
 - y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_0)_(server_0))
 - y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_0)_(server_1))
 = 0
_C29: x_(resnet50_conv1_relu_Relu)(server_1)
 - y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_1)_(server_0))
 - y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_1)_(server_1))
 = 0
_C3: x_(Conv__435)(server_0) + x_(Conv__435)(server_1) = 1
_C30: x_(resnet50_pool1_pad_Pad)(server_0)
 - y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_0)_(server_0))
 - y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_0)_(server_1))
 = 0
_C31: x_(resnet50_pool1_pad_Pad)(server_1)
 - y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_1)_(server_0))
 - y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_1)_(server_1))
 = 0
_C32: x_(resnet50_pool1_pool_MaxPool)(server_0)
 - y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_0)_(server_0))
 - y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_0)_(server_1))
 = 0
_C33: x_(resnet50_pool1_pool_MaxPool)(server_1)
 - y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_1)_(server_0))
 - y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_1)_(server_1))
 = 0
_C34: x_(resnet50_conv2_block1_1_conv_Conv2D)(server_0)
 - y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_0)_(server_0))
 - y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_0)_(server_1))
 = 0
_C35: x_(resnet50_conv2_block1_1_conv_Conv2D)(server_1)
 - y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_1)_(server_0))
 - y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_1)_(server_1))
 = 0
_C36: x_(resnet50_conv2_block1_1_relu_Relu)(server_0)
 - y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_0)_(server_0))
 - y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_0)_(server_1))
 = 0
_C37: x_(resnet50_conv2_block1_1_relu_Relu)(server_1)
 - y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_1)_(server_0))
 - y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_1)_(server_1))
 = 0
_C38: x_(resnet50_conv2_block1_2_conv_Conv2D)(server_0)
 - y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_0)_(server_0))
 - y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_0)_(server_1))
 = 0
_C39: x_(resnet50_conv2_block1_2_conv_Conv2D)(server_1)
 - y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_1)_(server_0))
 - y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_1)_(server_1))
 = 0
_C4: x_(resnet50_conv1_relu_Relu)(server_0)
 + x_(resnet50_conv1_relu_Relu)(server_1) = 1
_C40: x_(resnet50_conv2_block1_2_relu_Relu)(server_0)
 - y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_0)_(server_0))
 - y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_0)_(server_1))
 = 0
_C41: x_(resnet50_conv2_block1_2_relu_Relu)(server_1)
 - y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_1)_(server_0))
 - y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_1)_(server_1))
 = 0
_C42: x_(resnet50_conv1_conv_Conv2D__6)(server_0)
 - y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_0)_(server_0))
 - y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_1)_(server_0))
 = 0
_C43: x_(resnet50_conv1_conv_Conv2D__6)(server_1)
 - y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_0)_(server_1))
 - y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_1)_(server_1))
 = 0
_C44: x_(Conv__435)(server_0)
 - y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_0)_(server_0))
 - y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_1)_(server_0)) = 0
_C45: x_(Conv__435)(server_1)
 - y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_0)_(server_1))
 - y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_1)_(server_1)) = 0
_C46: x_(resnet50_conv1_relu_Relu)(server_0)
 - y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_0)_(server_0))
 - y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_1)_(server_0)) = 0
_C47: x_(resnet50_conv1_relu_Relu)(server_1)
 - y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_0)_(server_1))
 - y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_1)_(server_1)) = 0
_C48: x_(resnet50_pool1_pad_Pad)(server_0)
 - y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_0)_(server_0))
 - y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_1)_(server_0))
 = 0
_C49: x_(resnet50_pool1_pad_Pad)(server_1)
 - y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_0)_(server_1))
 - y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_1)_(server_1))
 = 0
_C5: x_(resnet50_pool1_pad_Pad)(server_0)
 + x_(resnet50_pool1_pad_Pad)(server_1) = 1
_C50: x_(resnet50_pool1_pool_MaxPool)(server_0)
 - y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_0)_(server_0))
 - y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_1)_(server_0))
 = 0
_C51: x_(resnet50_pool1_pool_MaxPool)(server_1)
 - y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_0)_(server_1))
 - y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_1)_(server_1))
 = 0
_C52: x_(resnet50_conv2_block1_1_conv_Conv2D)(server_0)
 - y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_0)_(server_0))
 - y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_1)_(server_0))
 = 0
_C53: x_(resnet50_conv2_block1_1_conv_Conv2D)(server_1)
 - y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_0)_(server_1))
 - y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_1)_(server_1))
 = 0
_C54: x_(resnet50_conv2_block1_1_relu_Relu)(server_0)
 - y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_0)_(server_0))
 - y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_1)_(server_0))
 = 0
_C55: x_(resnet50_conv2_block1_1_relu_Relu)(server_1)
 - y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_0)_(server_1))
 - y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_1)_(server_1))
 = 0
_C56: x_(resnet50_conv2_block1_2_conv_Conv2D)(server_0)
 - y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_0)_(server_0))
 - y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_1)_(server_0))
 = 0
_C57: x_(resnet50_conv2_block1_2_conv_Conv2D)(server_1)
 - y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_0)_(server_1))
 - y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_1)_(server_1))
 = 0
_C58: x_(resnet50_conv2_block1_2_relu_Relu)(server_0)
 - y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_0)_(server_0))
 - y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_1)_(server_0))
 = 0
_C59: x_(resnet50_conv2_block1_2_relu_Relu)(server_1)
 - y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_0)_(server_1))
 - y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_1)_(server_1))
 = 0
_C6: x_(resnet50_pool1_pool_MaxPool)(server_0)
 + x_(resnet50_pool1_pool_MaxPool)(server_1) = 1
_C60: x_(resnet50_conv2_block1_3_conv_Conv2D)(server_0)
 - y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_0)_(server_0))
 - y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_1)_(server_0))
 = 0
_C61: x_(resnet50_conv2_block1_3_conv_Conv2D)(server_1)
 - y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_0)_(server_1))
 - y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_1)_(server_1))
 = 0
_C62: x_(args_0_wrapper)(server_0) = 1
_C7: x_(resnet50_conv2_block1_1_conv_Conv2D)(server_0)
 + x_(resnet50_conv2_block1_1_conv_Conv2D)(server_1) = 1
_C8: x_(resnet50_conv2_block1_1_relu_Relu)(server_0)
 + x_(resnet50_conv2_block1_1_relu_Relu)(server_1) = 1
_C9: x_(resnet50_conv2_block1_2_conv_Conv2D)(server_0)
 + x_(resnet50_conv2_block1_2_conv_Conv2D)(server_1) = 1
Binaries
x_(Conv__435)(server_0)
x_(Conv__435)(server_1)
x_(args_0_wrapper)(server_0)
x_(args_0_wrapper)(server_1)
x_(resnet50_conv1_conv_Conv2D__6)(server_0)
x_(resnet50_conv1_conv_Conv2D__6)(server_1)
x_(resnet50_conv1_relu_Relu)(server_0)
x_(resnet50_conv1_relu_Relu)(server_1)
x_(resnet50_conv2_block1_1_conv_Conv2D)(server_0)
x_(resnet50_conv2_block1_1_conv_Conv2D)(server_1)
x_(resnet50_conv2_block1_1_relu_Relu)(server_0)
x_(resnet50_conv2_block1_1_relu_Relu)(server_1)
x_(resnet50_conv2_block1_2_conv_Conv2D)(server_0)
x_(resnet50_conv2_block1_2_conv_Conv2D)(server_1)
x_(resnet50_conv2_block1_2_relu_Relu)(server_0)
x_(resnet50_conv2_block1_2_relu_Relu)(server_1)
x_(resnet50_conv2_block1_3_conv_Conv2D)(server_0)
x_(resnet50_conv2_block1_3_conv_Conv2D)(server_1)
x_(resnet50_pool1_pad_Pad)(server_0)
x_(resnet50_pool1_pad_Pad)(server_1)
x_(resnet50_pool1_pool_MaxPool)(server_0)
x_(resnet50_pool1_pool_MaxPool)(server_1)
y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_0)_(server_0))
y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_0)_(server_1))
y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_1)_(server_0))
y_((Conv__435)_(resnet50_conv1_relu_Relu))((server_1)_(server_1))
y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_0)_(server_0))
y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_0)_(server_1))
y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_1)_(server_0))
y_((args_0_wrapper)_(resnet50_conv1_conv_Conv2D__6))((server_1)_(server_1))
y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_0)_(server_0))
y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_0)_(server_1))
y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_1)_(server_0))
y_((resnet50_conv1_conv_Conv2D__6)_(Conv__435))((server_1)_(server_1))
y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_0)_(server_0))
y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_0)_(server_1))
y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_1)_(server_0))
y_((resnet50_conv1_relu_Relu)_(resnet50_pool1_pad_Pad))((server_1)_(server_1))
y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_0)_(server_0))
y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_0)_(server_1))
y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_1)_(server_0))
y_((resnet50_conv2_block1_1_conv_Conv2D)_(resnet50_conv2_block1_1_relu_Relu))((server_1)_(server_1))
y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_0)_(server_0))
y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_0)_(server_1))
y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_1)_(server_0))
y_((resnet50_conv2_block1_1_relu_Relu)_(resnet50_conv2_block1_2_conv_Conv2D))((server_1)_(server_1))
y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_0)_(server_0))
y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_0)_(server_1))
y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_1)_(server_0))
y_((resnet50_conv2_block1_2_conv_Conv2D)_(resnet50_conv2_block1_2_relu_Relu))((server_1)_(server_1))
y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_0)_(server_0))
y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_0)_(server_1))
y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_1)_(server_0))
y_((resnet50_conv2_block1_2_relu_Relu)_(resnet50_conv2_block1_3_conv_Conv2D))((server_1)_(server_1))
y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_0)_(server_0))
y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_0)_(server_1))
y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_1)_(server_0))
y_((resnet50_pool1_pad_Pad)_(resnet50_pool1_pool_MaxPool))((server_1)_(server_1))
y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_0)_(server_0))
y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_0)_(server_1))
y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_1)_(server_0))
y_((resnet50_pool1_pool_MaxPool)_(resnet50_conv2_block1_1_conv_Conv2D))((server_1)_(server_1))
End
