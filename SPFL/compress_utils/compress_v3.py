"""
compress module v1
conjugate sgd and sgd algorthrim
need to do add fcn compress module
numpy based version parallel
time:20201028
"""

from collections import defaultdict
from copy import deepcopy

import numpy as np


def bmm_np(w_v,w_h):
    bmm = [np.matmul(w_v[i], w_h[i]) for i in range(w_v.shape[0])]
    cur_w = np.sum(np.stack(bmm, axis=0), axis=0)
    return cur_w


def kernel_merger(w_v,w_h):
    pass
    out_dim = w_h.shape[0]
    in_dim = w_v.shape[1]

    w = np.zeros((out_dim,in_dim,3,3))
    for out_index in range(out_dim):
        for in_index in range(in_dim):
            w_v_slice = w_v[:,in_index,:,:]
            w_h_slice = w_h[out_index,:,:,:]
            bmm = [np.matmul(w_v_slice[i],w_h_slice[i]) for i in range(w_v_slice.shape[0])]
            cur_w = np.sum(np.stack(bmm,axis=0),axis=0)
            w[out_index,in_index]  = cur_w
    return w


def kernel_merger_v1(w_v,w_h):
    pass
    out_dim = w_h.shape[0]
    in_dim = w_v.shape[1]
    d = w_v.shape[2]
    w = np.zeros((out_dim,in_dim,d,d))
    for out_index in range(out_dim):
        for in_index in range(in_dim):
            w_v_slice = np.squeeze(w_v[:,in_index,:,:],axis=-1).transpose(1,0)
            w_h_slice = np.squeeze(w_h[out_index,:,:,:],axis=1)
            w[out_index,in_index] = np.matmul(w_v_slice,w_h_slice)
    return w


def distance(w,w_v,w_h):
    """
    distance between w and recon kernel
    :param w:
    :param w_v:
    :param w_h:
    :return:
    """
    cur_merger = kernel_merger_v1(w_v,w_h)
    return np.sum((w-cur_merger)**2)


def distance_v1(w, w_v, w_h):
    """
    test new distance cal format base on w_v
    :param w:
    :param w_v:
    :param w_h:
    :return:
    """
    n = w.shape[0]
    c = w.shape[1]
    error = 0
    for c_index in range(c):
        cur_w = w[:, c_index, :, :]
        cur_w = cur_w.transpose(1, 2)
        cur_w = cur_w.reshape(cur_w.shape[0], -1)
        cur_v = np.squeeze(w_v[:, c_index, :, :], axis=-1).transpose(1, 0)
        cur_h = np.squeeze(w_h, axis=-2).transpose(1, 2, 0)
        cur_h = cur_h.reshape(cur_h.shape[0], -1)
        cur_re = np.matmul(cur_v, cur_h)
        error += np.sum((cur_w - cur_re) ** 2)
    return error



def distance_v2(w,w_v,w_h):
    """
    test new distance cal format base w_h
    :param w:
    :param w_v:
    :param w_h:
    :return:
    """
    n = w.shape[0]
    c = w.shape[1]
    error = 0
    for n_index in range(n):
        cur_w = w[n_index,:, :, :]
        cur_w = cur_w.transpose(2,1,0)
        cur_w = cur_w.reshape(cur_w.shape[0], -1)
        cur_v = np.squeeze(w_v,axis=-1).transpose(0,2,1)
        cur_v = cur_v.reshape(cur_v.shape[0],-1)
        cur_h = np.squeeze(w_h[n_index],axis=-2).transpose(1,0)
        cur_re = np.matmul(cur_h,cur_v)
        error = np.sum((cur_w - cur_re) ** 2)
    return error


def one_step_on_w_v_sgd(w, w_v, w_h,lr=0.01):
    """
    on sgd step on w_v
    :param w:
    :param w_v:
    :param w_h:
    :param lr:
    :return:
    """
    k = w_v.shape[0]
    c = w.shape[1]
    n = w.shape[0]
    for c_index in range(c):
        for k_index in range(k):
            d_w_v = np.zeros((w_v.shape[2],w_v.shape[3]))
            for n_index in range(n):
                w_v_slice = w_v[:, c_index, :, :]
                w_h_slice = w_h[n_index, :, :, :]
                cur_d_v = w[n_index, c_index] - bmm_np(w_v_slice,w_h_slice)
                cur_d_v = np.matmul(cur_d_v,np.transpose(w_h[n_index,k_index],axes=(1,0)))
                d_w_v -= cur_d_v
            w_v[k_index, c_index] -= lr * d_w_v
    return w_v


def one_step_on_w_h_sgd(w, w_v, w_h,lr=0.01):
    """
    one sgd step on w_h
    :param w:
    :param w_v:
    :param w_h:
    :param lr:
    :return:
    """
    k = w_v.shape[0]
    c = w.shape[1]
    n = w.shape[0]
    for n_index in range(n):
        for k_index in range(k):
            d_w_h = np.zeros((w_h.shape[2], w_h.shape[3]))
            for c_index in range(c):
                w_v_slice = w_v[:, c_index, :, :]
                w_h_slice = w_h[n_index, :, :, :]
                cur_d_h = w[n_index, c_index] - bmm_np(w_v_slice, w_h_slice)
                cur_d_h = np.matmul(np.transpose(w_v[k_index, c_index], axes=(1, 0)),cur_d_h)
                d_w_h -= cur_d_h
            w_h[n_index, k_index] -= lr * d_w_h
    return w_h


def conjugate_grad_w_v(w,w_v,w_h):
    """
    conjugate step on grad on w_v
    :return:
    """
    # print("org w is ")
    # print(w)
    # print("org w_v is")
    # print(w_v)
    # print("org w_h is")
    # print(w_h)
    n = w.shape[0]
    c = w.shape[1]
    k = w_v.shape[0]
    d = w.shape[2]

    update_step = k
    error = 0
    for c_index in range(c):
        #  cur_w with shape nd*d
        cur_w = w[:, c_index, :, :]
        cur_w = cur_w.transpose(1,2,0)
        cur_w = cur_w.reshape(cur_w.shape[0], -1)
        cur_w = cur_w.transpose(1,0)

        #  cur_h with shape nd*k
        cur_h = np.squeeze(w_h,axis=-2).transpose(1,2,0)
        cur_h = cur_h.reshape(cur_h.shape[0], -1)
        cur_h = cur_h.transpose(1,0)

        #  cur_v with shape k*d
        cur_v = np.squeeze(w_v[:, c_index, :, :],axis=-1)
        # cur_re = torch.matmul(cur_h, cur_v)
        # error += torch.sum((cur_w - cur_re) ** 2)
        # continue

        #  min fun if ||cur_h*cur_v - cur_w ||**2
        #  Q with shape is k*k
        Q = np.matmul(np.transpose(cur_h,axes=(1,0)),cur_h)
        #  update base on different d dim
        for d_index in range(d):
            #  cur_v_slice with shape k*1
            cur_v_slice = np.expand_dims(cur_v[:,d_index],axis=-1)
            #  cur_w_slice with shape nd*1
            cur_w_slice = np.expand_dims(cur_w[:,d_index],axis=-1)
            #  B with shape 1*k
            B = np.matmul(np.transpose(cur_h,axes=(1,0)),cur_w_slice)
            # print(cur_v_slice.shape,B.shape)
            for step in range(update_step):
                grad = np.matmul(Q,cur_v_slice) - B
                grad_T = np.transpose(grad,axes=(1,0))
                if step == 0:
                    d_dir = -grad
                else:
                    same_part_Q_d_dir = np.matmul(Q,d_dir)
                    beta = np.matmul(grad.transpose(1,0),same_part_Q_d_dir)\
                           /np.matmul(d_dir.transpose(1,0),same_part_Q_d_dir)
                    d_dir = -grad + beta*d_dir
                d_dir += 1e-10
                Q_m_dir = np.matmul(Q,d_dir)
                a_step = - (np.matmul(grad_T,d_dir))/(np.matmul(d_dir.transpose(1,0),Q_m_dir))
                # print("a step with shape {}".format(a_step.shape))
                # print(a_step)
                # print("d_dir with shape {}".format(d_dir.shape))
                # print(d_dir)
                cur_v_slice = cur_v_slice + a_step*d_dir
            cur_v[:,d_index] = cur_v_slice[:,0]
        w_v[:, c_index, :, :] = np.expand_dims(cur_v,axis=-1)
        # print(distance_v1(w,w_v,w_h))
    # print(w_v)
    # print(error)
    return w_v


def conjugate_grad_w_h(w,w_v,w_h):
    """
    conjugate step on grad on w_h
    :return:
    """
    n = w.shape[0]
    c = w.shape[1]
    k = w_v.shape[0]
    d = w.shape[2]

    update_step = k
    error = 0
    for n_index in range(n):
        #  cur w with shape cd*d
        cur_w = w[n_index,:,:,:]
        cur_w = cur_w.reshape(-1,cur_w.shape[-1])

        #  cur_v with shape cd*k
        cur_v = np.squeeze(w_v,-1).transpose(1,2,0)
        cur_v = cur_v.reshape(-1,cur_v.shape[-1])

        #  cur_w with shape k*d
        cur_h = np.squeeze(w_h[n_index],axis=1)
        # cur_re = torch.matmul(cur_v, cur_h)
        # error += torch.sum((cur_w - cur_re) ** 2)
        # continue
        #  min fun if ||cur_h*cur_v - cur_w ||**2
        #  Q with shape is k*k
        Q = np.matmul(np.transpose(cur_v, axes=(1, 0)), cur_v)
        #  update base on different d dim
        for d_index in range(d):
            #  cur_v_slice with shape k*1
            cur_h_slice = np.expand_dims(cur_h[:, d_index],axis=-1)
            #  cur_w_slice with shape nd*1
            cur_w_slice = np.expand_dims(cur_w[:, d_index],axis=-1)
            #  B with shape 1*k
            B = np.matmul(np.transpose(cur_v, axes=(1, 0)), cur_w_slice)
            # print(cur_v_slice.shape,B.shape)
            for step in range(update_step):
                grad = np.matmul(Q, cur_h_slice) - B
                grad_T = np.transpose(grad, axes=(1, 0))
                if step == 0:
                    d_dir = -grad
                else:
                    same_part_Q_d_dir = np.matmul(Q, d_dir)
                    beta = np.matmul(grad.transpose(1, 0), same_part_Q_d_dir) \
                           / np.matmul(d_dir.transpose(1, 0), same_part_Q_d_dir)
                    d_dir = -grad + beta * d_dir
                d_dir += 1e-10
                Q_m_dir = np.matmul(Q, d_dir)
                a_step = - (np.matmul(grad_T, d_dir)) / (np.matmul(d_dir.transpose(1, 0), Q_m_dir))
                # print("a step with shape {}".format(a_step.shape))
                # print(a_step)
                # print("d_dir with shape {}".format(d_dir.shape))
                # print(d_dir)
                cur_h_slice = cur_h_slice + a_step * d_dir
            cur_h[:, d_index] = cur_h_slice[:, 0]
        w_h[n_index] = np.expand_dims(cur_h,axis=1)
        # print(distance_v1(w, w_v, w_h))
        # print(w_v)
        # print(error)
    return w_h


def sgd_decompose(w,k):
    """
    decompose kernel w with shape N*C*D*D  to
    w_v with shape K*C*D*1 and w_h with shape N*K*1*D
    based on sgd solve
    :param w:model kernel para
    :param k:output channel of w_v
    :return:
    """
    N,C,D,D = w.shape
    w_v = np.random.randn(k,C,D,1)
    w_h = np.random.randn(N,k,1,D)
    for e in range(100):
        old_dis = distance(w,w_v,w_h)
        w_v = one_step_on_w_v_sgd(w,w_v,w_h)
        w_h = one_step_on_w_h_sgd(w,w_v,w_h)
        # w_v = conjugate_grad_w_v(w,w_v,w_h)
        # w_h = conjugate_grad_w_h(w,w_v,w_h)
        # print(distance(w, w_v, w_h))
        new_dis = distance(w,w_v,w_h)
        print("sgd solve step {},old dis is {:.4f},new dis is {:.4f}".format(e,old_dis,new_dis))
        if np.abs(new_dis - 0) < 0.00001 or np.abs(new_dis - old_dis) / old_dis < 0.001:
            break
    return w_v,w_h


def conjugate_sgd_decompose(w,k):
    """
    decompose kernel w with shape N*C*D*D  to
    w_v with shape K*C*D*1 and w_h with shape N*K*1*D
    based on conjugate sgd solve
    :param w:model kernel para
    :param k:output channel of w_v
    :return:
    """

    N, C, D, D = w.shape
    w_v = np.random.randn(k, C, D, 1)
    w_h = np.random.randn(N, k, 1, D)
    for e in range(100):
        old_dis = distance(w, w_v, w_h)
        # w_v = one_step_on_w_v_sgd(w, w_v, w_h)
        # w_h = one_step_on_w_h_sgd(w, w_v, w_h)
        w_v = conjugate_grad_w_v(w,w_v,w_h)
        w_h = conjugate_grad_w_h(w,w_v,w_h)
        # print(distance(w, w_v, w_h))
        new_dis = distance(w, w_v, w_h)
        # print("conjudate sgd solve step {},old dis is {:.6f},new dis is {:.6f}".format(e, old_dis, new_dis))
        # print(np.abs(new_dis-0))
        # print(np.abs(new_dis-old_dis)/old_dis)
        if np.abs(new_dis-0)<0.0001 or np.abs(new_dis-old_dis)/old_dis < 0.001:
            break
    return w_v, w_h


def reconstruct_para(w_v,w_h):
    """
    reconstruct kernel para based on w_v and w_h
    :param w_v:w_v with shape K*C*D*1
    :param w_h:w_h with shape N*C*1*D
    :return:reconstruct kernel w with shape N*C*D*D
    """
    out_dim = w_h.shape[0]
    in_dim = w_v.shape[1]
    kernel_size = w_v.shape[2]

    w = np.zeros((out_dim, in_dim, kernel_size, kernel_size))
    for out_index in range(out_dim):
        for in_index in range(in_dim):
            w_v_slice = np.squeeze(w_v[:, in_index, :, :], axis=-1).transpose(1, 0)
            w_h_slice = np.squeeze(w_h[out_index, :, :, :], axis=1)
            w[out_index, in_index] = np.matmul(w_v_slice, w_h_slice)
    return w


def quantization_fun(w):
    """
    quantize current kernel w based on natural compress algorithm
    :param w:model kernel para
    :return:w after quantization
    """
    w_sign = (((w >= 0).astype(np.float))*2-1)
    # w_sign = w>=0
    w = np.log2(np.abs(w+0.0000001))
    w = w.astype(np.int)
    w = w.astype(np.float)
    w = 2**w
    w = w*w_sign
    return w


def compress_fun(local_compress_module, local_update_model_grad_dict: dict):
    """
    parallel compress fun compress module is based on numpy
    :param local_compress_module:
    :param local_update_model_grad_dict:
    :return:
    """
    local_compress_grad_dict = local_compress_module.compress(local_update_model_grad_dict)
    local_recon_grad_dict = local_compress_module.reconstruct(local_compress_grad_dict)
    return local_recon_grad_dict


class CompressModule:

    def __init__(self,
                 com_solve_method="conjugate_sgd",
                 compress_ratio=4,
                 use_quan=True,
                 use_para_decompose=True):
        assert com_solve_method in ["sgd","conjugate_sgd"]
        self.compress_ratio = compress_ratio

        self.solve_method = com_solve_method
        if self.solve_method == "sgd":
            self.compress_solver = sgd_decompose
        elif self.solve_method == "conjugate_sgd":
            self.compress_solver = conjugate_sgd_decompose
        else:
            self.compress_solver = None

        self.reconstruct_solver = reconstruct_para

        self.quantization_solver = quantization_fun

        self.use_quantization = use_quan
        self.use_para_decompose = use_para_decompose

    def compress(self,model_dict):
        """
        compress current model dict
        :param model_dict:org model dict
        :return:
        """
        compress_model_dict = defaultdict(list)
        for key_name in model_dict.keys():
            # print("process  layer:",key_name)
            current_kernel_para = deepcopy(model_dict[key_name])
            if self.use_quantization:
                current_kernel_para = self.quantization_solver(current_kernel_para)
            if self.use_para_decompose:
                if "conv" in key_name and "weight" in key_name:
                    current_para_shape = current_kernel_para.shape
                    k_value = int((current_para_shape[0]*current_para_shape[1]*current_para_shape[2])/
                                  (current_para_shape[0]+current_para_shape[1])/self.compress_ratio)
                    if k_value <= 1:
                        k_value = 1
                    # print(k_value)
                    current_kernel_v_para,current_kernel_h_para = self.compress_solver(current_kernel_para,k_value)
                    current_kernel_para = [current_kernel_v_para,current_kernel_h_para]
                    # print(current_para_shape,current_kernel_v_para.shape,current_kernel_h_para.shape)
            compress_model_dict[key_name] = current_kernel_para
        # print(model_dict)
        # print(compress_model_dict)
        return compress_model_dict

    def reconstruct(self,model_dict):
        """
        reconstruct model para base on current model dict
        :param model_dict:a dict,for conv layer
                          its value contain two item w_v and w_h
        :return:
        """
        recon_model_dict = dict()
        for key_name in model_dict.keys():
            if self.use_para_decompose \
                    and "conv" in key_name \
                    and "weight" in key_name:
                cur_decom_para = deepcopy(model_dict[key_name])
                assert len(cur_decom_para) == 2
                w_v, w_h = cur_decom_para
                w = self.reconstruct_solver(w_v, w_h)
                recon_model_dict[key_name] = w
            else:
                recon_model_dict[key_name] = deepcopy(model_dict[key_name])
        return recon_model_dict


def test_v1():
    # SEED = 776
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # np.random.seed(SEED)
    w = np.random.randn(1,2,5,5)
    quan_w = quantization_fun(w)
    print("org w")
    print(w)
    print("quanzation w")
    print(quan_w)
    error = np.mean(np.abs(w - quan_w))
    print(error)
    w_v = np.random.randn(1, 1, 3, 1)
    w_h = np.random.randn(1, 1, 1, 3)
    # w = quan_w

    w = kernel_merger_v1(w_v,w_h)
    w_v, w_h = sgd_decompose(w,1)
    w_v, w_h = conjugate_sgd_decompose(w, 1)
    re_w = kernel_merger_v1(w_v, w_h)
    print(re_w)
    error = np.mean(np.abs(w - re_w))
    print(error)


def test_v2():

    import torch
    model_dict = torch.load('./cifar100_global_net_model.pt')

    for k, v in model_dict.items():
        model_dict[k] = v.cpu().numpy()
    # print(model_dict)
    conv1_weight = model_dict['conv2.weight']
    print(conv1_weight.shape)

    w = conv1_weight
    quan_w = quantization_fun(w)
    print("org w")
    print(w[0,0])
    print("quanzation w")
    print(quan_w[0,0])
    error = np.mean(np.abs(w - quan_w))
    print(error)
    # w_v = torch.randn((1, 1, 3, 1))
    # w_h = torch.randn((1, 1, 1, 3))
    w = quan_w

    # w = kernel_merger_v1(w_v,w_h)
    # sgd_decompose(w,1)
    w_v, w_h = conjugate_sgd_decompose(w, 4)
    re_w = kernel_merger_v1(w_v, w_h)
    print(re_w[0,0])
    error = np.mean(np.abs(w - re_w))
    print(error)


def test_v3():
    import torch
    model_dict = torch.load('./cifar100_global_net_model.pt')
    for k, v in model_dict.items():
        model_dict[k] = v.cpu().numpy()
    # print(model_dict)
    # for key_name in model_dict.keys():
    #     print("="*20)
    #     conv1_weight = model_dict[key_name]
    #     print("current layer is :{},para weight shape is :{}".format(key_name,conv1_weight.shape))
    #     print("org w")
    #     # print(conv1_weight[0, 0])
    #     print("do quanzation------>")
    #     w = conv1_weight
    #     quan_w = quantization_fun(w)
    #     print("quanzation w")
    #     # print(quan_w[0, 0])
    #     error = np.mean(np.abs(w - quan_w))
    #     print("error is:",error)
    #     if "conv" in key_name and "weight" in key_name:
    #         print("do compress------>")
    #
    #         w = quan_w
    #         w_v, w_h = conjugate_sgd_decompose(w, 17)
    #         re_w = kernel_merger_v1(w_v, w_h)
    #         # print(re_w[0, 0])
    #         error = np.mean(np.abs(w - re_w))
    #         print("error is:", error)
    # print(type(model_dict))
    import time
    start_time = time.time()
    compress_module = CompressModule(use_para_decompose=True,use_quan=False)
    compress_dict = compress_module.compress(model_dict)
    recon_dict = compress_module.reconstruct(compress_dict)
    end_time = time.time()
    print("use time is {}".format(end_time - start_time))

    for key_name in model_dict.keys():
        print("key name error is",key_name)
        print(recon_dict[key_name].shape,model_dict[key_name].shape)
        print(np.mean(np.abs(recon_dict[key_name]-model_dict[key_name])))


if __name__=="__main__":

    test_v3()
    pass