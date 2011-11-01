"""DataPath class specification"""
import sys
import time
import datetime
import urllib2
import subprocess
import getpass
import ast
from utils import file_tools as ft


def print_dictionary(dict_in, handle, key_list=None, prepend=""):
    r"""print a dictionary in a slightly nicer way
    `key_list` specifies the keys to print; None is to print all
    `prepend` is a string to prepend to each entry (useful to tag)
    """
    if key_list == None:
        key_list = dict_in.keys()

    for key in key_list:
        print >> handle, "%s%s: %s" % (prepend, key, repr(dict_in[key]))


def extract_split_tag(keylist, divider="_", ignore=None):
    r"""take a list like ['A_ok', 'B_ok'], and return ['A', 'B']
    list is made unique and sorted alphabetically
    anything in `ignore` is thrown out
    """
    taglist = []
    for key in keylist:
        taglist.append(key.split(divider)[0])

    taglist = list(set(taglist))

    if ignore is not None:
        taglist = [tag for tag in taglist if tag not in ignore]

    taglist.sort()
    return taglist


def unique_cross_pairs(list1, list2):
    r"""given ['A','B'], ['A','B'] return ['AB']"""
    retlist = []
    for ind1 in range(len(list1)):
        for ind2 in range(len(list2)):
            if ind2 > ind1:
                retlist.append((list1[ind1], list2[ind2]))

    return retlist


# TODO: write doctest for this
def cross_maps(map1key, map2key, noise_inv1key, noise_inv2key,
               map_suffix="_clean_map", noise_inv_suffix="_noise_inv",
               verbose=True):
    r"""Use the database to report all unique crossed map maps given map and
    noise_inv keys.
    """
    dbp = DataPath()
    retpairs = {}
    retpairslist = []

    (map1keys, map1set) = dbp.fetch(map1key, intend_read=True,
                                    silent=True)

    (map2keys, map2set) = dbp.fetch(map2key, intend_read=True,
                                    silent=True)

    (noise_inv1keys, noise_inv1set) = dbp.fetch(noise_inv1key,
                                        intend_read=True,
                                        silent=True)

    (noise_inv2keys, noise_inv2set) = dbp.fetch(noise_inv2key,
                                        intend_read=True,
                                        silent=True)

    map1tags = extract_split_tag(map1keys, ignore=['firstpass'])
    map2tags = extract_split_tag(map1keys, ignore=['firstpass'])
    noise_inv1tags = extract_split_tag(noise_inv1keys, ignore=['firstpass'])
    noise_inv2tags = extract_split_tag(noise_inv2keys, ignore=['firstpass'])

    if (map1tags != noise_inv1tags) or (map2tags != noise_inv2tags):
        print "ERROR: noise_inv and maps are not matched"
        sys.exit()

    if verbose:
        print "Using map1 tags %s and map2 tags %s" % (map1tags, map2tags)

    comblist = unique_cross_pairs(map1tags, map2tags)
    for (tag1, tag2) in comblist:
        pairname = "%s_with_%s" % (tag1, tag2)
        retpairslist.append(pairname)
        pairdict = {'map1': map1set[tag1 + map_suffix],
                    'noise_inv1': noise_inv1set[tag1 + noise_inv_suffix],
                    'map2': map2set[tag2 + map_suffix],
                    'noise_inv2': noise_inv2set[tag2 + noise_inv_suffix],
                    'tag1': tag1, 'tag2': tag2}

        if verbose:
            print "-"*80
            print_dictionary(pairdict, sys.stdout,
                            key_list=['map1', 'noise_inv1',
                                      'map2', 'noise_inv2',
                                      'tag1', 'tag2'])

        retpairs[pairname] = pairdict

    return (retpairslist, retpairs)


class DataPath(object):
    r"""A class to manage the data path database

    'load_pathdict' in __init__ downloads and executes the group-writable
    database specification and it executes and extracts the path database and
    file groups. __init__ also prints the user's relevant github status
    information for logging (associating a run with a code version).

    The database specification format can be found in that file's doc string:
    /cita/d/www/home/eswitzer/GBT_param/path_database.py
    or
    http://www.cita.utoronto.ca/~eswitzer/GBT_param/path_database.py

    Usage:
    # start the class, printing version information
    >>> datapath_db = DataPath()
    DataPath: opening URL
        http://www.cita.utoronto.ca/~eswitzer/GBT_param/path_database.py
    DataPath: opening URL
        http://www.cita.utoronto.ca/~eswitzer/GBT_param/hashlist.txt
    # checksums compiled: ... by ...
    DataPath: File database date/version ...
    DataPath: Run info: ... by ...
    git_SHA: ...
    git_blame: ...
    git_date: ...
    DataPath: ... files registered; database size in memory ...

    # pick index '44' of the 15hr sims
    >>> datapath_db.fetch("sim_15hr_beam", pick='44')
    (sim_15hr_beam) => .../simulations/15hr/sim_beam_044.npy: ...
    '.../simulations/15hr/sim_beam_044.npy'

    # get the 15hr sim path
    >>> datapath_db.fetch("sim_15hr_path")
    (sim_15hr_path) => .../simulations/15hr/: ...
    '.../simulations/15hr/'

    TODO: also allow dbs from local paths instead of URLs
    TODO: switch to ordered dictionaries instead of list+dictionary?
    TODO: code check that all files in the db exist, etc.
    TODO: make command line to regen file hash, web page
    TODO: make fetch handle file hashes and note updates

    Extensions to consider:
        -require writing to a log file; check exists; opt. overwrite
        -functions support links through e.g.
            15hrmap -> 15hrmap_run5_e1231231231_etc_Etc.
        -framework for data over http
    """

    # URL to the default path database
    _db_root = "http://www.cita.utoronto.ca/~eswitzer/GBT_param/"
    _db_file = "path_database.py"
    _db_url_default = _db_root + _db_file
    _hash_file = "hashlist.txt"
    _hash_url_default = _db_root + _hash_file

    def __init__(self, db_url=_db_url_default, hash_url=_hash_url_default,
                 skip_gitlog=False):
        r"""Load a file path specification and get basic run info
        """

        self._pathdict = {}       # the main path database
        self._hashdict = {}       # file hash database
        self._groups = {}         # dictionary of database key lists by group
        self._group_order = []    # order in which to print the groups
        self.version = "Empty"    # the database version
        self.db_url = db_url      # URL: database specification
        self.hash_url = hash_url  # URL: file hash specification
        self.gitlog = "Empty"     # git SHA and commit info

        # load the file database and git version info
        self.load_pathdict(db_url)
        self.load_hashdict(hash_url)
        self.check_groups()
        self.clprint("File database date/version " + self.version)

        # get user and code version information
        dateinfo = datetime.datetime.now()
        self.runinfo = (dateinfo.strftime("%Y-%m-%d %H:%M:%S"),
                        getpass.getuser())
        self.clprint("Run info: %s by %s" % self.runinfo)
        if not skip_gitlog:
            self.get_gitlog()
        self._db_size = (len(self._hashdict),
                        sys.getsizeof(self._pathdict) +
                        sys.getsizeof(self._hashdict))

        self.clprint("%d files registered; database size in memory = %s" %
                     self._db_size)

    def clprint(self, string_in):
        r"""print with class message; could extend to logger"""
        print "DataPath: " + string_in

    def load_pathdict(self, db_url, print_dbspec=False):
        r"""Load the parameter dictionary

        note that the class instance update()'s its dictionary, so subsequent
        calls of this function on different files will overwrite or augment
        dictionaries that have already been loaded.
        """
        self.clprint("opening URL " + self.db_url)

        resp = urllib2.urlopen(db_url)
        # urllib2 will die if this fails, but to be safe,
        if (resp.code != 200):
            print "ERROR: path database URL invalid (%s)" % resp.code

        path_db_spec = resp.read()
        if print_dbspec:
            print "-" * 80
            self.clprint("executing: ")
            print path_db_spec
            print "-" * 80

        # evaluate the database specification inside a dictionary
        bounding_box = {}
        code = compile(path_db_spec, '<string>', 'exec')
        exec code in bounding_box

        # extract only the useful database information
        self.version = bounding_box['version_tag']
        self._pathdict.update(bounding_box['pdb'])
        self._groups.update(bounding_box['groups'])
        self._group_order = bounding_box['group_order']

    def load_hashdict(self, hash_url):
        r"""Load the file hash library
        """
        self.clprint("opening URL " + self.hash_url)

        resp = urllib2.urlopen(hash_url)
        if (resp.code != 200):
            print "ERROR: path database URL invalid (%s)" % resp.code
        hash_spec = resp.read()
        hash_spec = hash_spec.split("\n")
        for entry in hash_spec:
            if len(entry) > 0:
                if (entry[0] == "#"):
                    print entry
                else:
                    parser = entry.split(": ")
                    filename = parser[0]
                    filename = filename.strip()
                    fileinfo = ast.literal_eval(parser[1])
                    self._hashdict[filename] = fileinfo

    def check_groups(self):
        r"""check that all the database keys are accounted for by groups
        """
        keylist_in_groups = []
        keylist = self._pathdict.keys()

        for item in self._groups:
            keylist_in_groups.extend(self._groups[item])

        for groupkey in keylist_in_groups:
            if groupkey not in keylist:
                print "ERROR: " + groupkey + " is not in the database"
                sys.exit()

        for dbkey in keylist:
            if dbkey not in keylist_in_groups:
                print "ERROR: " + dbkey + " is not in the group list"
                sys.exit()

    def print_db_item(self, dbkey, suppress_lists=90, silent=False):
        r"""print a database entry to markdown format
        'desc' and 'status' are requires keys in the file attributes
        suppress printing of lists of more than `suppress_lists` files
        `silent` only returns a string instead of printing
        """
        dbentry = self._pathdict[dbkey]
        retstring = "### `%s`\n" % dbkey
        retstring += "* __Description__: %s\n" % dbentry['desc']

        if 'status' in dbentry:
            retstring += "* __Status__: %s\n" % dbentry['status']

        if 'notes' in dbentry:
            retstring += "* __Notes__: %s\n" % dbentry['notes']

        if 'filelist' in dbentry:
            listindex = dbentry['listindex']
            retstring += "* __List index__: `" + repr(listindex) + "`\n"

            if (len(listindex) <= suppress_lists):
                retstring += "* __File list__:\n"
                for listitem in listindex:
                    filename = dbentry['filelist'][listitem]
                    if filename in self._hashdict:
                        retstring += "    * `%s`: `%s` `%s`\n" % \
                                    (listitem, filename, self._hashdict[filename])
                    else:
                        retstring += "    * `%s`: `%s`\n" % \
                                    (listitem, filename)

        if 'path' in dbentry:
            retstring += "* __Path__: `%s`\n" % dbentry['path']

        if 'file' in dbentry:
            filename = dbentry['file']
            if filename in self._hashdict:
                retstring += "* __File__: `%s` `%s`\n" % \
                             (filename, self._hashdict[filename])
            else:
                retstring += "* __File__: `%s`\n" % dbentry['file']

        if not silent:
            print retstring

        return retstring

    def print_path_db(self, suppress_lists=90, fileobj=None):
        r"""print all the files in the path database; note that it is a
        dictionary and so this is un-ordered. print_path_db_by_group prints the
        database items ordered by the file group specifications.

        You should only need to use this if the db groups are broken.

        Suppress printing of lists of more than `suppress_lists` files.
        If given a filename, this will write to a text file in markdown format.
        """
        print "-" * 80

        for dbkey in self._pathdict:
            dbstring = self.print_db_item(dbkey,
                                          suppress_lists=suppress_lists)
            print dbstring

        print "-" * 80

    def print_path_db_by_group(self, suppress_lists=90, fileobj=None):
        r"""print all the files in the path database ordered by group
        specification.

        Suppress printing of lists of more than `suppress_lists` files.
        If given a filename, this will write to a text file in markdown format.
        """
        print "-" * 80

        for groupname in self._group_order:
            print "%s\n%s\n" % (groupname, "-" * len(groupname))
            if fileobj:
                fileobj.write("****\n %s\n%s\n\n" % \
                              (groupname, "-" * len(groupname)))

            for dbkey in self._groups[groupname]:
                dbstring = self.print_db_item(dbkey,
                                              suppress_lists=suppress_lists)
                if fileobj:
                    fileobj.write(dbstring + "\n")

        print "-" * 80

    def get_gitlog(self):
        r"""parse the github status for this code version for logging
        """
        process = subprocess.Popen(["git", "log"], stdout=subprocess.PIPE)
        gitlog = process.communicate()[0]
        gitlog = gitlog.split("\n")
        gitdict = {}
        gitdict['SHA'] = " ".join(gitlog[0].split()[1:])
        gitdict['blame'] = " ".join(gitlog[1].split()[1:])
        gitdict['date'] = " ".join(gitlog[2].split()[1:])
        gitdict['note'] = " ".join(gitlog[4].split()[1:])
        self.gitlog = gitdict

        print_dictionary(self.gitlog, sys.stdout,
                         key_list=["SHA", "blame", "date"],
                         prepend="git_")


    def fetch(self, dbkey, pick=None, intend_read=False, intend_write=False,
              purpose="", silent=False):
        r"""The access function for this database class:
        Fetch the data for a requested key in the db.

        `pick` takes one index from a file list
        if the database entry is a file list, return a tuple of the index
        indices and a dictionary (the list orders the dictionary)

        if `intend_write` then die if the path does not exist or not writable
        if `intend_read` then die if the path does not exist
        `purpose` inputs a purpose for this file for logging
        `silent` does not print anything upon fetch unless error
        """
        dbentry = self._pathdict[dbkey]
        prefix = "%s (%s) " % (purpose, dbkey)

        if 'file' in dbentry:
            pathout = dbentry['file']
            ft.path_properties(pathout, intend_write=intend_write,
                               intend_read=intend_read, is_file=True,
                               prefix=prefix, silent=silent)

        if 'path' in dbentry:
            pathout = dbentry['path']
            ft.path_properties(pathout, intend_write=intend_write,
                               intend_read=intend_read, is_file=False,
                               prefix=prefix, silent=silent)

        if 'filelist' in dbentry:
            pathout = (dbentry['listindex'], dbentry['filelist'])
            if pick:
                pathout = pathout[1][pick]
                ft.path_properties(pathout, intend_write=intend_write,
                                   intend_read=intend_read, is_file=True,
                                   prefix=prefix, silent=silent)
            else:
                for item in pathout[0]:
                    filepath = pathout[1][item]
                    ft.path_properties(filepath, intend_write=intend_write,
                                       intend_read=intend_read, is_file=True,
                                       file_index=item, prefix=prefix,
                                       silent=silent)

        return pathout

    def generate_path_webpage(self):
        r"""Write out a markdown file with the file database
        """
        localpath = "/cita/d/www/home/eswitzer/GBT_param/"
        localdb = localpath + "path_database.py"
        dbwebpage = localpath + "path_database.txt"

        print "writing markdown website to: " + dbwebpage
        fileobj = open(dbwebpage, "w")

        fileobj.write("Data path DB\n============\n\n")
        fileobj.write("* specified by local path: `%s` and URL `%s`\n" %
                      (localdb, self.db_url))
        fileobj.write("* file hash table specified by `%s`\n" % self.hash_url)
        fileobj.write("* website, checksums compiled: %s by %s\n" % \
                       self.runinfo)
        fileobj.write("* %d files registered; database size in memory = %s\n" %
                     self._db_size)
        self.print_path_db_by_group(suppress_lists=90, fileobj=fileobj)

        fileobj.close()

    def generate_hashtable(self):
        r"""Write out the file hash table
        """
        localpath = "/cita/d/www/home/eswitzer/GBT_param/"
        dbhashpage = localpath + "hashlist.txt"

        print "writing file hash list to: " + dbhashpage
        hashobj = open(dbhashpage, "w")
        hashdict = {}
        hashlist = []

        hashobj.write("# checksums compiled: %s by %s\n" % \
                       self.runinfo)

        for groupname in self._group_order:
            for dbkey in self._groups[groupname]:
                dbentry = self._pathdict[dbkey]

                if 'filelist' in dbentry:
                    listindex = dbentry['listindex']
                    for listitem in listindex:
                        filename = dbentry['filelist'][listitem]
                        hashdict[filename] = ft.hashfile(filename)
                        hashlist.append(filename)
                        print "%s: %s" % (filename, hashdict[filename])

                if 'file' in dbentry:
                    filename = dbentry['file']
                    hashdict[filename] = ft.hashfile(filename)
                    hashlist.append(filename)
                    print "%s: %s" % (filename, hashdict[filename])

        print_dictionary(hashdict, hashobj, key_list=hashlist)
        hashobj.close()


if __name__ == "__main__":
    import doctest

    # generate the path db markdown website
    datapath_db = DataPath()
    datapath_db.generate_hashtable()
    datapath_db.generate_path_webpage()

    # run some tests
    OPTIONFLAGS = (doctest.ELLIPSIS |
                   doctest.NORMALIZE_WHITESPACE)
    doctest.testmod(optionflags=OPTIONFLAGS)
