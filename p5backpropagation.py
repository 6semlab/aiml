import numpy as np
x = np.array(([9,2],[1,5],[3,6]),dtype=float)
y = np.array(([92],[86],[89]),dtype=float)
c = np.amax(x,axis=0)

x=x/c
y=y/100
print(x)
print(y)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_grad(x):
    return x*(1-x)


epoch=1000
eta=0.15
input_neuron=2
hidden_neuron=3
output_neuron=1
np.random.seed(2)
wh=np.random.uniform(low=-0.05,high=0.05,size=(input_neuron,hidden_neuron))
print(wh)
bh=np.random.uniform(low=-0.05,high=0.05,size=(1,hidden_neuron))
print(bh)
wout=np.random.uniform(low=-0.05,high=0.05,size=(hidden_neuron,output_neuron))
print(wout)
bout=np.random.uniform(low=-0.05,high=0.05,size=(1,output_neuron))
print(bout)



for i in range(epoch):
    h_ip=np.dot(x,wh)+bh
#     print(h_ip)
    h_act=sigmoid(h_ip)
    o_ip=np.dot(h_act,wout)+bout
    output = sigmoid(o_ip)
    
    #error at o/p layer
    Eo=y-output
    outgrad=sigmoid_grad(output)
    d_output = Eo*outgrad
#     print("ITERATION ",i," Error",d_output)
    
    #error at hidden layer
    Eh=np.dot(d_output,wout.T)
    hiddengrad = sigmoid_grad(h_act)
    d_hidden = Eh*hiddengrad
    wout += np.dot(h_act.T,d_output)*eta
    wh += np.dot(x.T,d_hidden)*eta
#     print("output",output)

    sum_error=0
    for j in range(len(y)):
        print(output[j],y[j])
        sum_error+=((output[j]-y[j])**2)
    sum_error/=2
    print(f'Epoch {i} error {sum_error}')
print("Normalized input: \n"+str(x))
print("Actual output: \n"+str(y))
print("Predicted output: \n",output)
