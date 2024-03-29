import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

def plot_gradient(theta1, J):
	#print('Plotting gradient...')
	fig, ax = plt.subplots(figsize=(10,4.8))
	ax.scatter(theta1, J, c='black', s=40, lw=0, alpha = 0.5)
	ax.set_xlabel(r'$\theta_1$')
	ax.set_ylabel('RSS')
	ax.set_title('Função de custo')
	plt.show()





def step_gradient(b_current, m_current, points, learning_rate):
	#gradient descent
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))

	for i in range(1,len(points)):
		x = points[i,0]
		y = points[i,1]
		b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)
	return [new_b, new_m]


def compute_error_for_given_points(b, m, points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		totalError += (y - (m * x + b))**2
	return totalError / float(len(points))


def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
	b = initial_b
	m = initial_m

	theta1 = [m]
	J = [compute_error_for_given_points(b, m, points)]

	for i in range(num_iterations):
		b, m = step_gradient(b, m, np.array(points), learning_rate)
		#print('RSS: %0.2f' %(compute_error_for_given_points(b, m, points)))
		theta1.append(m)
		J.append(compute_error_for_given_points(b, m, points))
	plot_gradient(theta1, J)
	return [b, m]

def main():
	#points = np.genfromtxt('data.csv', delimiter=',')
	points = np.genfromtxt('income.csv', delimiter=',')
	#Hiperparametros
	#learning_rate = 0.0035
	learning_rate = 0.001

	#y = mx + b
	initial_b = 0
	initial_m = 0
	num_iterations = 1000

	[b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	#print('b = %0.5f, m = %0.5f' %(b,m))
	print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_given_points(b, m, points)))


if __name__ == '__main__':
	main()