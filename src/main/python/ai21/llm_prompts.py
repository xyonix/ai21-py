SENTIMENT_SENTENCE_LABELING = dict(preface=
"""
Label the following sentence by sentiment:
Frankly, my dear, I don't give a damn

Labels:
Indifferent, Apathetic, Neutral

##

Label the following sentence by sentiment:
I'm gonna make him an offer he can't refuse

Labels:
Threatening, Scheming 

##

Label the following sentence by sentiment:
The flower that blooms in adversity is the most rare and beautiful of all

Labels:
Hopeful, Inspirational, Thought Invoking, Optimistic

##

Label the following sentence by sentiment:
""",
end='\n\nLabels:\n')


TERSE_SUMMARY = dict(prompt=
"""
Summarize the following paragraph:
Information overload (also known as infobesity, infoxication, information anxiety, and information explosion) is the difficulty in understanding an issue and effectively making decisions when one has too much information (TMI) about that issue, and is generally associated with the excessive quantity of daily information. The term "Information overload" was first used in Bertram Gross' 1964 book, The Managing of Organizations, and was further popularized by Alvin Toffler in his bestselling 1970 book Future Shock. Speier et al. (1999) said that if input exceeds the processing capacity, information overload occurs, which is likely to reduce the quality of the decisions.
Summary:
Information overload is the difficulty in understanding an issue and effectively making decisions when one has too much information about that issue.

##

Summarize the following paragraph:
Like almost all software, an API needs to reflect the needs of the humans who interact with it. An API is somewhat different from a GUI or other user interface because it interacts with a programmer rather than directly with the end user. However, the principles of designing for the user still apply: you must think of the background knowledge brought by the programmers and the constraints under which they are operating, such as the bandwidth of mobile devices.
Summary:
An API is similar to a GUI, but it should reflect the needs of the programmer and the constraints of the device.

##

Summarize the following paragraph:
Computers that can read and write are here, and they have the potential to fundamentally impact daily life. The future of human-machine interaction is full of possibility and promise, but any powerful technology needs careful deployment. The joint statement below represents a step towards building a community to address the global challenges presented by AI progress, and we encourage other organizations who would like to participate to get in touch.
Summary:
Computers that can read and write are here, and they can fundamentally impact daily life. A community is being built to address the challenges presented by AI progress.

##

Summarize the following paragraph:
""",
end='\nSummary:\n')