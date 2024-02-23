# Exercise 2c

### **The Mandelbrot set**

The Mandelbrot set is generated on the complex plane $\mathbb{C}$  by iterating the complex function $f_c(z)$ whose form is
$$
f_c(z) = z^2 + c \label{eq:mandelbrot}
$$
for a complex point $c=x+iy$ and starting from the complex value $z=0$ so to obtain the series
$$
z_0 = 0,\, z_1 = f_c(0),\, z_2 = f_c(z_1),\, \dots,\, f_c^n(z_{n-1})
$$

The $Mandelbrot\, Set\, \mathcal{M}$ is defined as the set of complex points $c$ for which the above sequence is bounded. It may be proved that once an element $i$ of the series  is more distant than 2 from the origin, the series is then unbounded.
Hence, the simple condition to determine whether a point $c$ is in the set $\mathcal{M}$ is the following
$$
\left| z_n = f_c^n(0)\right| < 2\;\; \text{or}\;\; n > I_{max}
\label{eq:condition}
$$
where $I_{max}$ is a parameter that sets the maximum number of iteration after which you consider the point $c$ to belong to $\mathcal{M}$ (the accuracy of your calculations increases with $I_{max}$, and so does the computational cost).

Given a portion of the complex plane, included from the bottom left corner $c_L = x_L + iy_L$ and the top right one $c_R = x_R + iy_R$, an image of $\mathcal{M}$, made of $n_x \times n_y$ "pixels" can be obtained deriving, for each point $c_i$ in the plane, the sequence $z_n(c_i)$ to which apply the condition $\eqref{eq:condition}$, where
$$
\begin{aligned}
c_i &= (x_L + \Delta x) + i(y_L + \Delta y) \\
 \Delta x &= (x_R - x_L) / n_x \\
 \Delta y &= (y_R - y_L) / n_y\,.
\end{aligned}
$$
In practice, you define a 2D matrix `M` of integers, whose entries `[j][i]` correspond to the image's pixels. What pixel of the complex plane $\mathbb{C}$ corresponds to each element of the matrix depends obviously on the parameters $(x_L, y_L), (x_R, y_R), n_x, \text{ and } n_y$.

Then you give to a pixel `[j][i]` either the value of 0, if the corresponding $c$ point belongs to $\mathcal{M}$, or the value $n$ of the iteration for which
$$
\left| z_n\left(c\right) \right| > 2
$$
($n$ will saturate to $I_{max}$).

This problem is obviously embarrassingly parallel, for each point can be computed independently of each other and the most straightforward implementation would amount to evenly subdivide the plane among concurrent processes (or threads). However, in this way you will possibly find severe imbalance problems because the $\mathcal{M}$'s inner points are computationally more demanding than the outer points, the frontier being the most complex region to be resolved.

#### Requirements:

1. Your code must accept $I_{max}, c_L, c_R, n_x \text{ and } n_y$ as arguments. Specifically, the compilation must produce an executable whose execution has a proper default behavior and accept arguments as follows:
   `./executable n_x  n_y  x_L  y_L  x_R  y_R  I_max`

2. The size of integers of your matrix `M` shall be either `char` (1 byte; $I_{max} = 255$) or `short int` (2 bytes; $I_{max}=65535$).

3. Your code must produce a unique output file. You may use MPI I/O if you choose to implement the MPI+OpenMP version, **directly producing an image file** using the very simple format `.pgm` that contains a grayscale image.
   You find a function to do that, and the relative simple usage instructions, in Appendix I at the end of this page.
   In this way, you may check in real-time and by eye whether the output of your code is meaningful.

4. You have to determine the strong and weak scaling of your code. If you are developing a hybrid MPI+OpenMP code (option 2 at the beginning of this file), the scalings must be conducted as follows:

   * _OMP scaling_ : run with a single MPI task and increase the number of OMP threads
   * _MPI scaling_ : run with a single OMP thread per MPI task and increase the number of MPI tasks. Use at least two nodes, preferably four.

   The corresponding plots will be part of your report.
