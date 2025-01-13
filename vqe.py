import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import scipy

def energy_objective_vqe(param_values,single_param,double_param,ansatz,energy_observable):
    expectaon_calculation=tfq.layers.Expectation(
    differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.001),
        )
    param1={str(single_param[i]):param_values[i] for i in range(len(single_param))}
    param2={str(double_param[i]):param_values[i+len(single_param)] for i in range(len(double_param))}
    param1.update(param2)
    intput_points=np.array(list(param1.values())).reshape(1,-1).astype(np.float32)
    value_tensor=tf.convert_to_tensor(intput_points,dtype=np.float32)
    with tf.GradientTape() as g:
        g.watch(value_tensor)
        output=expectaon_calculation(
            ansatz,
            operators=energy_observable,
            symbol_names=[*param1.keys()],
            symbol_values=value_tensor,
        )
    analytical_gradient=g.gradient(output,value_tensor)
    return output.numpy()[0],analytical_gradient.numpy()[0]
def optimize_amplitude(energy_objective_vqe,initial_amplitudes,ts,td,anstaz,h_mol):
    result = scipy.optimize.minimize(energy_objective_vqe,
                                    initial_amplitudes,
                                    args=(ts,td,anstaz,h_mol),
                                    method="BFGS",
                                    jac=True,
                                    tol=1e-4,
                                    options={'maxiter':100,'disp': True}) 
    return result