# Main Idea
We will build a RBM machine learning model, to recommend a movie to an user, based on what he clamis to like it. The idea is that users rate the movies and with this information our model will try to find correlations and then predict a movie which the user may like it.
* We will use PyTorch for this implementation. 

# Results
We have got excellent results, with an average of 93%.

# Intuition of RBM machines --> Model based on probability. Really interesting

# PyTorch
For our implementation, we are going to use PyTorch. Is a dinymic library, which is really good when using Unsupervised learning
and also for some supervised learning, like RNN and CNN.
* It's faster
* And better in some ways, than Keras

# Boltzmann Machines - Stochastic model. Used when no much data available. Uses probability for state
* Are undirected models -- like a complete graph
* Visiable nodes & Hidden Nodes(Same) --> They work together, generanting data
* Even though we do get a fixed data, they don't just expect data, they generate data
* It generates different states, even without an especific data
* Thus, we use or data to help the model to ajust the weights accordinally, for our speficifc problem


# Energy-Based Models
* Probability distribution
* Based on the data given Boltzmann Machines ajusts the weights, to find the lowerst energy state(by probability)	

# Restricted Boltzmann Machines(RBM)
* Same idea of Boltzmann Machines. However, here visible nodes do not connect to each other, and so does hidden nodes.
* This is because for a full connected model, it becomes a very high cost model to compute.
* It generates states at all times
* During training, we ajust the sistem to be good to our example

# Contrastive  --> Used to train Boltzmann Machines --< Most used
	# Gibbs Sampling
		* How to ajust weights? Not backpropagation anymore. Uses Contrastive
		* The hidden Nodes reconstructs the Visible nodes, at every interaction
			* Ajusting the weights, to make sure about the activation for that visible node
		* The idea is "similar" to backprogapation. Trying to find the best place to land, with less energy(minumum error in backpropagation)
			* However, we change the weights to change our curve(minimum energy possibable)
				* This way for each iteration
		* How to do it in a short cut? (From geofry hinton)
			* We use CDs. Basically, from few states, we can already see which direction we are going
			* And therefore, how to ajust the curve in such way to get the low energy as possible. 
			* Avoiding going step by step

# Deep Belief Networks(DBN) 
* More than one RBM together.
* It's really powerfull, but really complex


# Aplications:
* Audio/Music creation
* Recommender(a movie for example)
	* For a especific user, we feed with information about movies he has seen, and movies he hasn't. 
		* Based on the movies he has seen, it analysies the features(hidden nodes) that was activated
		* Then, tries to find some correlation
		* the goal then is to sugest a movie.
