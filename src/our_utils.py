import itertools
import torch
import pickle
import glob
import math

with open('all_fp_dic.pkl','rb') as f:
    all_fp_dic = pickle.load(f)
    
def find_bias(w):
  w_abs = torch.abs(w)
  max_w_abs = torch.max(w_abs).item()
  max_exponent = math.log2(max_w_abs)
  max_exponent = math.floor(max_exponent)
  return max_exponent
  
      
    
def binaryVec2number(num_vec ,dyn_bias ,input_order_bits = {'sign': 0, 'exponent': 1, 'mantissa': 4}, ieee= True, max_bias = True):
    """
    Convert a binary vector to a number
    :param num_vec: binary vector after quantization
    :param input_order_bits: bits order in the vector
    :return: number that this vector represent
    """
    sign_bits = input_order_bits['sign']
    exponent_bits = input_order_bits['exponent']
    mantissa_bits = input_order_bits['mantissa']
    # get the sign bit
    # if a sign bit is really used
    if exponent_bits != 0:
        sign = int(num_vec[sign_bits:exponent_bits],2)
    else:
        sign = 0
    # get the exponent bits
    exponent = int(num_vec[exponent_bits:mantissa_bits],2)

    num_of_mantissa_bits = len(num_vec[mantissa_bits:])
    num_of_exponent_bits = mantissa_bits-exponent_bits
    if max_bias:
        bias = 2**(num_of_exponent_bits) - 1 - dyn_bias
    else:
        bias =  2**(num_of_exponent_bits) - 1  # 2**(num_of_exponent_bits - 1) - 1 - was the defualt but we changed to our defualt (with max bias until 1 representation)
    # if eponent is all ones - it's NAN
    max_exponent = 2**(num_of_exponent_bits) - 1
    # get the mantissa bits
    mantissa = int(num_vec[mantissa_bits:],2)
    if exponent == 0:
        return (-1)**sign * 2**(exponent - bias + 1) *  (mantissa * 2**(-num_of_mantissa_bits))  
    # In this function we'll return 0 instead of NAN - will not affect quantization
    elif exponent == max_exponent and ieee:
        return  0
    return (-1)**sign * 2**(exponent - bias) * (1 + mantissa * 2**(-num_of_mantissa_bits))

def quantWeights(fp_bits, w, dyn_bias, input_order_bits = {'sign': 0, 'exponent': 1, 'mantissa': 4}, ieee = True, max_bias = True):
    """
    Convert a weight tensor to a quantisezed tensor
    :param fp_bits: number of bits in quantisezed fp
    :param w: weights tensor
    :param input_order_bits: bits order in the vector
    :return: quantisezed weights tensor up and down
    """
    sign_bits = input_order_bits['sign']
    exponent_bits = input_order_bits['exponent']
    mantissa_bits = input_order_bits['mantissa']
    # key for pkl dictionary
    fp_input_order_bits_tup = (fp_bits, sign_bits,exponent_bits,mantissa_bits, ieee, max_bias, dyn_bias)
    # number of possible fp numbers
    tensor_size = 2**fp_bits
    if fp_input_order_bits_tup in all_fp_dic:
        s_all_fp_numbers = all_fp_dic[fp_input_order_bits_tup]
    
    else:
        all_fp_numbers = torch.zeros(tensor_size).cuda()
        # all possible binary combinations
        all_vecs = list(itertools.product([0, 1], repeat = fp_bits))
        # calculate for each binary the right fp
        for i,vec in enumerate(all_vecs):
            vec_str = "".join(map(str,list(vec)))
            all_fp_numbers[i] = (binaryVec2number(vec_str, dyn_bias, input_order_bits, ieee, max_bias))
        # sort 
        s_all_fp_numbers,_ = torch.sort(all_fp_numbers)
        all_fp_dic[fp_input_order_bits_tup] = s_all_fp_numbers
        with open('all_fp_dic.pkl', 'wb') as f:
            pickle.dump(all_fp_dic, f)
    #print(s_all_fp_numbers)       
    quant_idx = torch.bucketize(w,s_all_fp_numbers)
    idx_limits_max  = torch.where(quant_idx > (tensor_size - 1), torch.ones_like(quant_idx) * (-1), torch.zeros_like(quant_idx))
    idx_limits_min  = torch.where(quant_idx < 1, torch.ones_like(quant_idx), torch.zeros_like(quant_idx))
    quant_idx = quant_idx + idx_limits_max + idx_limits_min
    quant_up = s_all_fp_numbers[quant_idx]
    quant_down = s_all_fp_numbers[quant_idx-1]
    return quant_up, quant_down

def closestQuant(fp_bits, w, input_order_bits = {'sign': 0, 'exponent': 1, 'mantissa': 4}, ieee = True, max_bias = True):
    """
	Calculate the closest quantization for a weight tensor
	:param w: weights tensor
	:param fp_bits: number of bits in quantisezed fp
	:param input_order_bits: bits order in the vector
	:return: closest quantisezed weights tensor
    """
    dyn_bias = find_bias(w)
    q_up, q_down = quantWeights(fp_bits, w,dyn_bias, input_order_bits  ,ieee , max_bias)
    diff_up = torch.abs(w - q_up)
    diff_down = torch.abs(w - q_down)
    # compare l1 errors and return mask tensor of the closest quantization
    mask = torch.where(diff_up < diff_down, torch.ones_like(diff_up), torch.zeros_like(diff_down))
    return q_up * mask + q_down*(1-mask)
    
def clacQuantError(w,w_quant):
    """
    Calculate l1 and l2 error for quantization
    :param w: weghits tensor
    :param w_quant: quantisezed weights tensor
    :return: l1 error, l2 error
    """ 
    l1tensor = torch.abs(w-w_quant)
    l1error = l1tensor.sum() / torch.numel(l1tensor)
    l2tensor = torch.square(w-w_quant)
    l2error = l2tensor.sum() / torch.numel(l2tensor)
    return l1error, l2error



#fp_bits = 5
#w = torch.tensor([[0.005,0.8], [-0.008,-0.002]]).cuda()
#q_up, q_down = quantWeights(fp_bits, w, input_order_bits = {'sign': 0, 'exponent': 1, 'mantissa': 4},ieee=False)
#print("round up:\n",q_up)
#print("round down:\n",q_down)
#print("round closets:\n",closestQuant(fp_bits, w, input_order_bits = {'sign': 0, 'exponent': 1, 'mantissa': 4}, max_bias=False))
#print("round closets:\n",closestQuant(fp_bits, w, input_order_bits = {'sign': 0, 'exponent': 1, 'mantissa': 4}, max_bias=True))
#l1errUp, l2errUp = clacQuantError(w,q_up)
#l1errDown, l2errDown = clacQuantError(w,q_down)
#print("quant error for Up: l1: ", l1errUp, " l2: ", l2errUp)
#print("quant error for Down: l1: ", l1errDown, " l2: ", l2errDown)
