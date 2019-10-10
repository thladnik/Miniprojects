from Translation_Stimulus_Spherical_Arena import *

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author Tim Hladnik

frametime = .05
dur = 5.

stim = Stimulus()


# Create translation stimuli for different spatial frequencies and velocities
for sf in np.linspace(1./180, 5./180, 5):

    # Create a pattern
    pattern = Pattern.Bars(sf=sf)

    for v in np.linspace(.1, 1., 5):

        background = createTranslationStimulus(stim.verts,
                                               pattern=pattern, duration=dur, v=.0, frametime=frametime)
        pos_transl = createTranslationStimulus(stim.verts,
                                               pattern=pattern, duration=dur, v=v, frametime=frametime)
        neg_transl = createTranslationStimulus(stim.verts,
                                               pattern=pattern, duration=dur, v=-v, frametime=frametime)

        phase = applyMasks(stim.verts, background,
                           ['transl_stripe_symm', np.pi / 4, np.pi / 4, pos_transl],
                           ['transl_stripe_symm', -np.pi / 4, np.pi / 4, neg_transl],
                           )
        stim.addPhase(phase)

stim.display(frametime)

stim.saveAs('example01')