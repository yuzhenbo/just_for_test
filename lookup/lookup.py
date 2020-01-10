import gc
import ops as ops
import timeit
import torch
from tqdm import tqdm

class Lookup_table:
    def __init__(self,OPS:dict,input_shapes:list,name='ops.txt',need_calc=True):
        self.ops=OPS
        self.input_shapes=input_shapes
        self.cnt_layers=len(input_shapes)
        self.name=name
        if need_calc:
            self.lat=self.compute_latency()
            self.to_file()
        else:
            try:
                self.lat=self.from_file()
            except Exception as e:
                self.lat=self.compute_latency()
                self.to_file()
    def compute_latency(self):
        latency_table_layer_by_ops = [{} for i in range(self.cnt_layers)]
        LATENCY_BATCH_SIZE = 1
        cnt_of_runs = 50

        for layer_id in range(self.cnt_layers-1):
            for op_name in self.ops:
                op = self.ops[op_name](self.input_shapes[layer_id],
                                       self.input_shapes[layer_id+1]).cuda()
                input_sample = torch.randn((LATENCY_BATCH_SIZE, self.input_shapes[layer_id][0],self.input_shapes[layer_id][1],self.input_shapes[layer_id][2])).cuda()
                globals()['op'], globals()['input_sample'] = op, input_sample
                total_time = timeit.timeit('output = op(input_sample)', setup="gc.enable()",
                                           globals=globals(), number=cnt_of_runs)
                # measured in micro-second
                latency_table_layer_by_ops[layer_id][op_name] = total_time / cnt_of_runs / LATENCY_BATCH_SIZE * 1e6
        return latency_table_layer_by_ops
    def to_file(self):
        with open(self.name,'w') as f:
            for op_name in self.ops:
                f.write(op_name)
                f.write('|')
                for i in range(self.cnt_layers-1):
                    f.write(str(self.lat[i][op_name]))
                    f.write('|')
                f.write('\n')
    def from_file(self):
        latency_table_layer_by_ops = [{} for i in range(self.cnt_layers)]
        with open(self.name,'r') as f:
            for op_name in self.ops:
                l = f.readline().strip().split('|')
                if op_name!=l[0]:
                    print('Mismatch!{} and {}.'.format(l[0],op_name))
                    raise AssertionError()
                #assert op_name==l[0]
                for i in range(self.cnt_layers-1):
                    latency=float(l[i+1])
                    latency_table_layer_by_ops[i][op_name] = latency
        return latency_table_layer_by_ops
    def get(self):
        s=''
        ss=[]
        for op_name in self.ops:
            s=op_name+'|'
            for i in range(self.cnt_layers - 1):
                s+=str(self.lat[i][op_name])
                s+='|'
            ss.append(s)
        return ss