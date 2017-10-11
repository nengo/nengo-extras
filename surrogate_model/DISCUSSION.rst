************
Discussion Notes
************

This note is to update/discuss the progress of the project

Allen 10/11/2017
=============

The core functionalities have been implemented by Jamie we have a working prototype! I think there are 3 types of further work to be done.

    1. Filling in the pieces - implementing more interpolation methods, and extending algorithm for higher dimension

    2. Software engineering - we need to discuss how this model might be used with Nengo. For example, do we always want to apply surrogate model to an entire ``Network`` for do we want to support approximating just an ``Ensemble``? (the paper suggests to approximate a "population" and I wasn't 100% sure). How do we want users to use this? Depending on these questions we might have to restructure the code.

    3. Reach for the stars (?) - there are many optimizations guidelines that Bryan suggests, which will probably be experimental. (mostly found in section 5.1 of the paper - GPU, handling higher dimension, etc).

My immediate work will be on (1), to get myself comfortable with the idea. We should concurrently do (2) so there's less code to rewrite. (3) will be easier once (1) is done. I predict the most work might be in (2) (as always is when writing code)

Goal for this next week: Get most of (1) done (the skeleton is built so I just have to fill in the algorithms), Begin (2) and figure out what we need to do.


