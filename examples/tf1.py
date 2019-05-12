import tensorflow as tf
import numpy as np
import sklearn


x1 = tf.constant(5)
x2 = tf.constant(6)

#result = x1 * x2
result = tf.multiply(x1, x2)
print(result)

# session = tf.Session()
# print(session.run(result))
# session.close()

with tf.Session() as session:
    output = session.run(result)
    print(output)
print(output)
#print(session.run(result))

####################################################################################################



