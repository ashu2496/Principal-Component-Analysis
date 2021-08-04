import os, glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


"""
Name: Patel, Ashit

"""

CELEBA_DIRPATH = 'celebA_dataset'
N_HEIGHT = 78
N_WIDTH = 78
N_TRAIN = 850

def get_eigenfaces(eigenvalues, eigenvectors, k):
  """
    Sorts the eigenvector by eigenvalues
    Returns the projection matrix (eigenfaces)

    faces_centered : N x d vector
      mean subtracted faces
    eigenvalues : 1 x d vector
      eigenvalues
    eigenvectors : d x d vector
      eigenvectors
    k : int
      number of eigenfaces to produce

    returns d x k vector
  """
  # f(s,v) -> w by selecting top k
  order = np.argsort(eigenvalues)[::-1] # sorting v on basis of s
  eigenvectors = eigenvectors[:, order]
  eigenvectors = eigenvectors[:, 0:k] #selcting top k
  return eigenvectors
  
def project(faces, faces_mean, eigenfaces):
  """
    Returns the projected faces (lower dimensionality)

    faces : N x d vector
      images of faces
    faces_mean : 1 x d vector
      per pixel average of images of faces
    eigenfaces : d x k vector
      projection matrix

    returns N x k vector
  """
  b = faces - faces_mean  # b = x - mu
  z = np.matmul(b, eigenfaces)  # z = b*w
  return z
  

def reconstruct(faces_projected, faces_mean, eigenfaces):
  """
    Returns the reconstructed faces (back-projected)

    faces_projected : N x k vector
      faces in lower dimensions
    faces_mean : 1 x d vector
      per pixel average of images of faces
    eigenfaces : d x k vector
      projection matrix

    returns N x d vector
  """
  # x_hat = z*w.T + mu
  x_hat = np.matmul(faces_projected, eigenfaces.T) + faces_mean
  return x_hat

def synthesize(eigenfaces, variances, faces_mean, k=50, n=25):
  """
    Synthesizes new faces by sampling from the latent vector distribution

    eigenfaces : d x k vector
      projection matrix
    variances : 1 x d vector
      variances
    faces_mean : 1 x d vector

    returns synthesized faces
  """
  # sampling n samples ,default 25
  variances = variances[:k] # selecting top k
  eigenfaces = eigenfaces[:, 0:k] # selecting top k
  z = np.random.normal(0, np.real(np.sqrt(variances)), (n, np.shape(variances)[0])) # Normalization through Guassian Distribution
  x_hat = np.matmul(z, eigenfaces.T) + faces_mean   # x_hat = z * w
  return x_hat


def mean_squared_error(x, x_hat):
  """
    Computes the mean squared error

    x : N x d vector
    x_hat : N x d vector

    returns mean squared error
  """
  # 1/N sum(x-x_hat)^2
  mse=np.mean((x-x_hat)**2)
  return mse


def plot_eigenvalues(eigenvalues):
  """
    Plots the eigenvalues from largest to smallest

    eigenvalues : 1 x d vector
  """
  # plot eigenvalues
  fig = plt.figure()
  fig.suptitle('Eigenvalues Versus Principle Components')
  plt.plot(eigenvalues)
  plt.show()


def visualize_reconstructions(faces, faces_hat, n=4):
  """
    Creates a plot of 2 rows by n columns
    Top row should show original faces
    Bottom row should show reconstructed faces (faces_hat)

    faces : N x k vector
      images of faces
    faces_hat : 1 x d vector
      images reconstructed faces
  """
  fig = plt.figure()
  fig.suptitle('Real Versus Reconstructed Faces')
  # (N x d) -> (N x sqrt(d) x sqrt(d))
  faces = np.reshape(faces,(np.shape(faces)[0],int(np.sqrt(np.shape(faces)[1])),-1))
  # (N x d) -> (N x sqrt(d) x sqrt(d))
  faces_hat = np.reshape(faces_hat,(np.shape(faces_hat)[0],int(np.sqrt(np.shape(faces_hat)[1])),-1))
  faces_hat = np.real(faces_hat)
  # plot first n faces, default 4
  for i in range(2*n):
    ax=fig.add_subplot(2,n,i+1)
    if i < n:
      ax.imshow(faces[i, ...], cmap='gray')
    else:
      ax.imshow(faces_hat[i-n, ...], cmap='gray')
  plt.show()


def plot_reconstruction_error(mses, k):
  """
    Plots the reconstruction errors

    mses : list
      list of mean squared errors
    ks : list
      list of k used
  """
  # plotting k VS mses
  fig = plt.figure()
  fig.suptitle('Reconstruction Error')
  plt.plot(k, np.real(mses))
  plt.show()


def visualize_eigenfaces(eigenfaces):
  """
    Creates a plot of 5 rows by 5 columns
    Shows the first 25 eigenfaces (principal components)
    For each dimension k, plot the d number values as an image

    eigenfaces : d x k vector
  """
  # (d x k) -> (k x sqrt(d) x sqrt(d))
  eigenfaces = np.reshape(eigenfaces.T,(np.shape(eigenfaces.T)[0],int(np.sqrt(np.shape(eigenfaces.T)[1])),-1))
  eigenfaces = np.real(eigenfaces)
  # plot eigen faces
  fig = plt.figure()
  fig.suptitle('Top 25 Eigenfaces')
  n=np.shape(eigenfaces)[0] # n = k, number of faces, i.e. 25 in this case
  size = int(np.ceil(np.sqrt(n)))
  for i in range(n):
    ax=fig.add_subplot(size, size, i+1)
    ax.imshow(eigenfaces[i, ...], cmap='gray')
  plt.show()

def visualize_synthetic_faces(faces):
  """
    Creates a plot of 5 rows by 5 columns
    Shows the first 25 synthetic faces

    eigenfaces : N x d vector
  """
  # (N x d) -> (N x sqrt(d) x sqrt(d))
  faces = np.reshape(faces, (np.shape(faces)[0],int(np.sqrt(np.shape(faces)[1])),-1))
  faces = np.real(faces)
  # plot synthetic faces
  fig = plt.figure()
  fig.suptitle('Synthetic Faces')
  n=np.shape(faces)[0]  # n = k, number of faces, i.e. 25 in this case
  for i in range(n):
    ax=fig.add_subplot(5,5,i+1)
    ax.imshow(faces[i, ...], cmap='gray')
  plt.show()


if __name__ == '__main__':
  # Load faces from directory
  face_image_paths = glob.glob(os.path.join(CELEBA_DIRPATH, '*.jpg'))

  print('Loading {} images from {}'.format(len(face_image_paths), CELEBA_DIRPATH))
  # Read images as grayscale and resize from (128, 128) to (78, 78)
  faces = []
  for path in face_image_paths:
    im = Image.open(path).convert('L').resize((N_WIDTH, N_HEIGHT))
    faces.append(np.asarray(im))
  faces = np.asarray(faces) # (1000, 78, 78)
  # Normalize faces between 0 and 1
  faces = faces/255.0

  print('Vectorizing faces into N x d matrix')
  # TODO: reshape the faces to into an N x d matrix
  faces=np.reshape(faces,(np.shape(faces)[0],-1))  # reshaping to (1000,6084)

  print('Splitting dataset into {} for training and {} for testing'.format(N_TRAIN, faces.shape[0]-N_TRAIN))
  faces_train = faces[0:N_TRAIN, ...]
  faces_test = faces[N_TRAIN::, ...]

  print('Computing eigenfaces from training set')
  # TODO: obtain eigenfaces and eigenvalues
  mu_train = np.mean(faces_train, axis=0) # calculating mean of faces
  b_train = faces_train - mu_train  # calculating b = faces - mean
  c = np.dot(b_train.T, b_train) / (b_train.shape[0]) # calculating variance = (b^2)/N
  s, v = np.linalg.eig(c) # getting eigenvalues and eigenvectors
  eigenfaces = get_eigenfaces(s, v, np.shape(v)[0]) # calculating all eigenfaces

  print('Plotting the eigenvalues from largest to smallest')
  # TODO: plot the first 200 eigenvalues from largest to smallest
  s = sorted(s, reverse=True) # sorting eigenvalue from largest to smallest
  plot_eigenvalues(np.real(s[:200]))  # plotting top 200 eigenvalues

  print('Visualizing the top 25 eigenfaces')
  # TODO: visualize the top 25 eigenfaces
  visualize_eigenfaces(eigenfaces[:,0:25])  # showing top 25 eigenfaces

  print('Plotting training reconstruction error for various k')
  # TODO: plot the mean squared error of training set with
  # k=[5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]
  k=[5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]
  mses_train=[]
  for i in k:
    faces_projected_train = project(faces_train, mu_train, eigenfaces[:, 0:i])  # projecting faces using top k eigenfaces
    faces_reconstructed_train = reconstruct(faces_projected_train, mu_train, eigenfaces[:, 0:i])  # reconstructing faces
    mses_train = np.append(mses_train, mean_squared_error(faces_train, faces_reconstructed_train))  # calculating and storing mean squared error for each k
  plot_reconstruction_error(mses_train, k)  # plotting mse VS k

  print('Reconstructing faces from projected faces in training set')
  # TODO: choose k and reconstruct training set
  k = 200
  faces_projected_train = project(faces_train, mu_train, eigenfaces[:, 0:k])  # project training set faces with selected k = 800
  faces_reconstructed_train = reconstruct(faces_projected_train, mu_train, eigenfaces[:, 0:k])  # reconstruct projected faces
  
  # TODO: visualize the reconstructed faces from training set
  visualize_reconstructions(faces_train, faces_reconstructed_train) # show reconstructed faces

  print('Reconstructing faces from projected faces in testing set')
  # TODO: reconstruct faces from the projected faces
  faces_projected_test = project(faces_test, mu_train, eigenfaces[:, 0:k])  # project testing set faces with selected k = 800
  faces_reconstructed_test = reconstruct(faces_projected_test, mu_train, eigenfaces[:, 0:k])  # reconstruct projected faces

  # TODO: visualize the reconstructed faces from training set
  visualize_reconstructions(faces_test, faces_reconstructed_test) # show reconstructed faces

  print('Plotting testing reconstruction error for various k')
  # TODO: plot the mean squared error of testing set with
  # k=[5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]
  k=[5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]
  mses_test=[]
  for i in k:
    faces_projected_test = project(faces_test, mu_train, eigenfaces[:, 0:i])  # projecting faces using top k eigenfaces
    faces_reconstructed_test = reconstruct(faces_projected_test, mu_train, eigenfaces[:, 0:i])  # reconstructing faces
    mses_test = np.append(mses_test, mean_squared_error(faces_test, faces_reconstructed_test))  # calculating and storing mean squared error for each k
  plot_reconstruction_error(mses_test, k) # plotting mse VS k

  print('Creating synethic faces')
  # TODO: synthesize and visualize new faces based on the distribution of the latent variables
  # you may choose another k that you find suitable
  faces_synthesized = synthesize(eigenfaces, s, mu_train, 300, 25)    # synthesize faces
  visualize_synthetic_faces(faces_synthesized)  # show synthetic faces
