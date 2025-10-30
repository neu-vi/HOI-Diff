import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]
        
    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'seq_name' in notnone_batches[0]:
        seq_name = [b['seq_name']for b in notnone_batches]
        cond['y'].update({'seq_name': seq_name})
    
    if 'obj_points' in notnone_batches[0]:
        obj_points = collate_tensors([b['obj_points']for b in notnone_batches])

        if len(obj_points.shape) < 3:
            obj_points = obj_points.unsqueeze(0)
        cond['y'].update({'obj_points': obj_points})  #  this part is changed for prompt-based generation

    if 'obj_bps' in notnone_batches[0]:
        obj_bps = collate_tensors([b['obj_bps']for b in notnone_batches])
        cond['y'].update({'obj_bps': obj_bps})  #  this part is changed for prompt-based generation


    return motion, cond


# an adapter to our collate func
def t2hoi_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'seq_name': b[7],
        'obj_points': torch.tensor(b[8]).float(),
        'obj_bps': torch.tensor(b[9]).float(),
    } for b in batch]
    return collate(adapted_batch)



def afford_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)

    motion = databatchTensor
    cond = {'y': {'mask': None}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})


    if 'seq_name' in notnone_batches[0]:
        seq_name = [b['seq_name']for b in notnone_batches]
        cond['y'].update({'seq_name': seq_name})
    
    if 'obj_points' in notnone_batches[0]:
        obj_points = collate_tensors([b['obj_points']for b in notnone_batches])
        if len(obj_points.shape) < 3:
            obj_points = obj_points.unsqueeze(0)
        cond['y'].update({'obj_points': obj_points})  #  this part is changed for prompt-based generation

    # if 'obj_faces' in notnone_batches[0]:
    #     obj_faces = [b['obj_faces']for b in notnone_batches]
    #     obj_faces = torch.as_tensor(obj_normals)
    #     if len(obj_faces.shape) < 3:
    #         obj_faces = obj_faces.unsqueeze(0)
    #     cond['y'].update({'obj_faces': obj_faces})  #  this part is changed for prompt-based generation
    
    return motion, cond


# an adapter to our collate func
def t2m_contact_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[5],
        'seq_name': b[6],
        'obj_points': torch.tensor(b[7]).float(),
        # 'obj_faces':b[8]
    } for b in batch]
    return afford_collate(adapted_batch)

