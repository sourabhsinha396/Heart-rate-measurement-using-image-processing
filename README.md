# Heart-rate-measurement-using-image-processing
Heart rate measurement using image processing using OpenCV library of Python

This focusses on using the OpenCV library of Python to find the approximate heart rate.
We know that everytime heart pumps in blood in our Arteries blood pressure increases and so do the intensity of Red channel(others too)in 
our face and
other areas of our body.
** The first step involves finding the face so that we can concentrate on it,for this we have used the Haar Cascade method and for this an XML 
file is needed which is provided in the repository.

** The second step is to acquire all the red,green,and blue channel individually and concentrating on any one channel.
** The third step is to plot the variation of say Red channel and acquire its plot,and then we need to apply peak detection techniques.
The number of peaks(Maxima or Minima) gives us the approximate heart rate.

