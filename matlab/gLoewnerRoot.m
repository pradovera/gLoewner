function folder = gLoewnerRoot()
%GLOEWNERROOT   Root directory of gLoewner installation.
%   FOLDER = GLOEWNERROOT() returns a string that is the name of the
%   directory where gLoewner is installed.
    folder = fileparts(which('gloewnerroot'));
end