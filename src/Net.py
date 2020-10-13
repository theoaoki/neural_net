from Node import Node
import Function, random

def Error_d(actual_y, expected_y):
  return 2*(actual_y - expected_y)

node_1 = Node([random.random()*2-1, random.random()*2-1],
              random.random()*2 - 1, Function.Linear)

A = random.random()*2 - 1
B = random.random()*2 - 1
C = random.random()*2 - 1

print("A = " + str(A) + ", " +
      "B = " + str(B) + ", " +
      "C = " + str(C))

STEP = 0.1
print("Original weights: " + str(node_1.w))
print("Original bias: " + str(node_1.b))

input_batch = []
output_batch = []
for i in range(100):
  x = random.random()*2 - 1
  y = random.random()*2 - 1
  input_batch.append([x, y])
  output_batch.append(A*x + B*y + C)

for i in range(100):
  net_out = node_1.output(input_batch[i])

  #update weights and bias
  node_1.grad_w(input_batch[i], Error_d(net_out, output_batch[i]), STEP)
  node_1.grad_b(Error_d(net_out, output_batch[i]), STEP)

print()
print("Final weights: " + str(node_1.w))
print("Final bias: " + str(node_1.b))
