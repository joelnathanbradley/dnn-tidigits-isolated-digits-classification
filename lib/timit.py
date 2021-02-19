
import os
import re

class Timit:

    def __init__(self, basedir):
        """
        Management of TIMIT corpus located at directory basedir
        """

        self.corpusdir = dict()
        # Loop across train/test and woman/man creating appropriate entries
        for datatype in ('train', 'test'):
            self.corpusdir[datatype] = os.path.join(basedir, datatype)

        # Build dictionary with class name to index map
        self.categories = dict()
        ordzero = ord('0')  # encoding for zero
        for idx in range(1,10):
            self.categories[chr(idx + ordzero)] = idx
        self.categories['o'] = self.categories['O'] = 0   # "Oh" is class 0
        self.categories['z'] = self.categories['Z'] = 10  # "Zero" is class 10

        # class index to label
        self.class_labels = ('oh', '1', '2', '3', '4', '5', '6',
                             '7', '8', '9', 'zero')

    def get_class_labels(self):
        """
        get_class_labels()
        :return:  List of labels corresponding to class indices
        """
        return self.class_labels

    def get_filenames(self, type, include_augment_data=False, ext='.wav'):
        """
        :param type: train|test
        :param gender: gender subdirectory
        :param ext: type of file to look for
        :return: list of files
        """

        # Set target directory
        dir = self.corpusdir[type]

        filelist = []
        # Traverse dir recursively.  At each iteration, root
        # contains the current base directory, dirs are the subdirectories,
        # and files are the files at that level.
        for root, dirs, files in os.walk(dir):
            if not include_augment_data and "augmented_data" in root:
                continue
            # Process files in this root
            for f in files:
                # Check extension in a case independent manner
                _, file_ext = os.path.splitext(f)
                file_ext.lower()
                if file_ext == ext:
                    filelist.append(os.path.join(root, f))  # got one!

        return filelist

    # Python regular expressions, see https://docs.python.org/3/howto/regex.html
    # Match any single digit TIMIT file; that starts with 1-9, o, O, z, or Z
    # Followed by a single letter allowing us to distinguish between multiple versions
    # of the same speech.
    # Example m = re_timit.match("za.wav")
    # m.group("Class") is z.  m is None if there is no match
    re_timit = re.compile("(?P<Class>[0-9zZoO])[a-z].*")

    def filename_to_class(self, files):
        """
        Convert filenames to class
        :param files: list of filenames
        :return: Returns numeric category of file.  Assumes files may be part of a full
           path and consist of a digit or the letters o or z followed by one or more
           letters and a file extension.  Categories are 0-9 for oh, one, two, ...
           or 10 for zero.

           e.g. .../train/woman/ac/8a.wav

        """



        errors = []  # list of files with problems
        classvalues = []   # List of categories
        for idx in range(len(files)):
            _, fname = os.path.split(files[idx])  # pull out filename without path
            m = self.re_timit.match(fname)
            if m:
                # File is a TIMIT file,
                classvalues.append(self.categories[m.group("Class")])
            else:
                errors.append(idx)

        if len(errors):
            raise ValueError("Unable to parse files at positions: "%(
                "%d, ".join(errors)))

        return classvalues

