i'd like to implement an advanced multi-sampling tool into this project. it will have multiple phases and allow me to accuratley sample an instrument library 

here is what i need you to do: 

step 1 : generate a new midi file called analyze_instrument.midi 
    - this file will sample all 88 notes at all 127 velocities (1-127) at a rapid rate, at 120 bpm, each note will be held for 1/32 note worth of time and there will be 1/32 note worth of silence before the next note is triggered (so hold = 0.0625 and release = 0.0625) 
    - i will then export the audio created by feeding my target instrument this midi file into a .flac for you to analyze
    - you only need to do this once, so no need to include it in the implementation, just create a file that matches this spec that i will use for every instrument going forwards

step 2 : spectrum analyze every note at every velocity : 
    - the goal of this process is to detect sample switching in the instrument being sampled. so for each individual note, there will be multiple velocity points at which the sample switches and the timbre changes, your goal is to identify the velocity at which the sample changes - this will manifest as a change in frequency response (fast forrier over that 32nd note) from one velocity to another. NOTE: the instrument samples WILL change dB level but not necessarily timbre, you should NOT count a change in dB level as a sample switch, only timbre. When I listen to all 127 velocities of a note by ear, I can identify the sample switches, you should be able to do the same as well. Your process will be something like this

    for (note = 0, note < 88, note++) {
        for (velocity = 2, velocity < 127, velocity++) {
            compare(fast_forrier(note(velocity),note(velocity-1))) > tolerance {
                sample switch detected @ velocity !
            }
        }
    }

step 3: generate a new midi file, similar to @archive/generate_session.py (this uses an old representation of the medata) to sample our instrument at ONLY the velocity switch points. So essentially sample all 88 notes, for each note sample you will play the note at all the velocities at which the sample switched. You also need to sample the full range of velocities, so always include velocity = 1 and 127 in the midi file. There will be a few variables in the script that can be changed:

    - sustain_hold = (length in seconds of sustain hold, default = 10)
    - sustain_release = (length in seconds of sustain release time, default = .5)
    - sample_release = true | false (turn on if i want to sample the release)
    - release_hold = (length in seconds of release sample hold, default = .5)
    - release_release = (length in seconds of release sample release time, default = 1.0)

    Now, the big difference from our previous implementation is I plan to hold the sustain AND release samples all in one flac file! So for each note and velocity in the midi, you will first have the sustain sample note (sustain_hold,sustain_release) followed by the release sample note (release_hold, release_release)

step 4: generate a input.toml file in the format of @input_template.toml that exactly matches the midi file generated in step 3.


LASTLY:

The final outcome of your work should be

1. A midi file like described in step 1.
2. A script that performs step 2, step 3, and step 4. By taking the midi file as input, and outputting the sampler.midi and the input.toml file


PLEASE PLAN YOUR WORK, AND ASK ALL THE QUESTIONS YOU NEED TO UNDERSTAND HOW TO IMPLEMENT THIS