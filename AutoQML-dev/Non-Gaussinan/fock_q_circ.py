import torch
import pennylane as qml

n_wires = 8
dev = qml.device('strawberryfields.fock', wires=n_wires, cutoff_dim=4)

def cartesian_to_polar(real, imag):
    r = torch.sqrt(real**2 + imag**2)
    phi = torch.atan2(imag, real)
    return r, phi

def pennylane_encoding(x, cfg, params, reupload_layer=None):
    n_wires = cfg.circuit.wires
    
    if cfg.training.paradigm == "unsupervised":
        for wire in range(n_wires):
            eta = x[wire, 0]
            pt_frac = x[wire, 1]
            phi = x[wire, 2]
            
            qml.Displacement(pt_frac, phi, wires=wire)
            qml.Squeezing(eta, 0.0, wires=wire)
            
    else: 
        for wire in range(n_wires):
            eta = x[wire, 0]
            pt = x[wire, 1]
            phi = x[wire, 2]

            var_pt_w = params["enc_var_pt"][reupload_layer, wire]
            var_eta_w = params["enc_var_eta"][reupload_layer, wire]
            var_phi_w = params["enc_var_phi"][reupload_layer, wire]
            
            d_real = var_eta_w * eta
            d_imag = var_pt_w * pt
            d_r, d_phi = cartesian_to_polar(d_real, d_imag)
            qml.Displacement(d_r, d_phi, wires=wire)
            
            s_real = var_phi_w * torch.cos(phi)
            s_imag = var_phi_w * torch.sin(phi)
            s_r, s_phi = cartesian_to_polar(s_real, s_imag)
            qml.Squeezing(s_r, s_phi, wires=wire)

def pennylane_bs_substrate(params, pairs, bs_set_index):
    k = 0
    for w1, w2 in pairs:
        theta = params["bs_theta"][bs_set_index, k]
        phi = params["bs_phi"][bs_set_index, k]
        qml.Beamsplitter(theta, phi, wires=[w1, w2])
        k += 1

def pennylane_lg_substrate(params, n_wires, lg_set_index, lg_type):
    if lg_type == "none" or lg_type is None:
        return
        
    for w in range(n_wires):
        for letter in lg_type:
            if letter == "d":
                r, phi = cartesian_to_polar(
                    params["lg_d_real"][lg_set_index, w], 
                    params["lg_d_imag"][lg_set_index, w]
                )
                qml.Displacement(r, phi, wires=w)
            elif letter == "s":
                r, phi = cartesian_to_polar(
                    params["lg_s_real"][lg_set_index, w], 
                    params["lg_s_imag"][lg_set_index, w]
                )
                qml.Squeezing(r, phi, wires=w)
            elif letter == "r":
                phi_scalar = params["lg_r_phi"][lg_set_index, w]
                qml.Rotation(phi_scalar, wires=w)

def pennylane_cx_substrate(params, pairs, cx_set_index):
    k = 0
    for w1, w2 in pairs:
        s = params["cx_s"][cx_set_index, k]
        qml.CX(s, wires=[w1, w2])
        k += 1

@qml.qnode(dev, interface="torch")
def qae_qnode(x_single, cfg, params, pairs):
    pennylane_encoding(x_single, cfg, params)
    
    bs_idx, lg_idx, cx_idx = 0, 0, 0
    
    for layer in range(cfg.circuit.layers):
        for block in cfg.circuit.layer_structure:
            if block == "bs":
                pennylane_bs_substrate(params, pairs, bs_idx)
                bs_idx += 1
            elif block == "lg":
                pennylane_lg_substrate(params, cfg.circuit.wires, lg_idx, cfg.circuit.lg_substrate_type)
                lg_idx += 1
            elif block == "cx":
                pennylane_cx_substrate(params, pairs, cx_idx)
                cx_idx += 1
                
    return (
        qml.probs(wires=cfg.circuit.trash_modes), 
        [qml.expval(qml.NumberOperator(w)) for w in range(cfg.circuit.wires)]
    )

@qml.qnode(dev, interface="torch")
def cvvc_qnode(x_single, cfg, params, pairs):
    n_wires = cfg.circuit.wires
    pennylane_encoding(x_single, cfg, params, reupload_layer=0)
    
    bs_idx, lg_idx = 0, 0
    
    for layer in range(cfg.circuit.layers):
        for block in cfg.circuit.layer_structure:
            if block == "bs":
                pennylane_bs_substrate(params, pairs, bs_idx)
                bs_idx += 1
            elif block == "lg":
                pennylane_lg_substrate(params, n_wires, lg_idx, cfg.circuit.lg_substrate_type)
                lg_idx += 1
                
        if cfg.circuit.reupload and layer < cfg.circuit.layers - 1:
            pennylane_encoding(x_single, cfg, params, reupload_layer=layer + 1)

    return (
        qml.probs(wires=list(range(n_wires))),
        [qml.expval(qml.NumberOperator(w)) for w in range(n_wires)]
    )

def QAE_loss_pennylane(cfg, x_batch, params, pairs):
    B = x_batch.shape[0]
    
    batch_F = []
    batch_n = []
    
    for b in range(B):
        probs, n = qae_qnode(x_batch[b], cfg, params, pairs)
        
        F = probs[0] 
        logF = torch.log(F + 1e-12)
        
        batch_F.append(F.unsqueeze(0))
        batch_n.append(torch.stack(n).unsqueeze(0))

    F_tensor = torch.cat(batch_F)
    logF_tensor = torch.log(F_tensor + 1e-12)
    n_tensor = torch.cat(batch_n)

    if cfg.training.loss_fn == "F":
        return -torch.expm1(logF_tensor), F_tensor, n_tensor
    elif cfg.training.loss_fn == "logF":
        return -logF_tensor, F_tensor, n_tensor
    elif cfg.training.loss_fn == "logF/n":
        return -logF_tensor / len(cfg.circuit.trash_modes), F_tensor, n_tensor


def CVVC_loss_pennylane(cfg, x_batch, params, pairs):
    B = x_batch.shape[0]
    
    batch_F = []
    batch_n = []
    
    for b in range(B):
        probs, n = cvvc_qnode(x_batch[b], cfg, params, pairs)
        batch_F.append(probs[0].unsqueeze(0)) 
        batch_n.append(torch.stack(n).unsqueeze(0))

    F_tensor = torch.cat(batch_F)
    logF_tensor = torch.log(F_tensor + 1e-12)
    n_tensor = torch.cat(batch_n)

    if cfg.training.loss_fn == "F":
        return -torch.expm1(logF_tensor), n_tensor

    elif cfg.training.loss_fn == "logF":
        raw_score = -logF_tensor
        logit = params["sup_logloss_scale"] * raw_score + params["sup_logloss_shift"]
        return logit, n_tensor

    elif cfg.training.loss_fn == "logF/n":
        raw_score = -logF_tensor / cfg.circuit.wires
        logit = params["sup_logloss_scale"] * raw_score + params["sup_logloss_shift"]
        return logit, n_tensor
