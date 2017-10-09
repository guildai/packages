# Guild Index

This is the package source repository for Guild Index.

Each sub-directory represents a package. Refer to the package
`README.md` for details.

This index is under active development and must be installed manually
for use with Guild. See [Installing the index](#install-the-index) for
more information.

## Installing the index

Guild commands that access an index (i.e. `install`, `search`, and
`update`) will use `~/.guild/index` by default. If you install the
index in a different location, use the `--index` option to specify
that location when running these commands.

The steps below assume you are installing the index in the default
location (`~/.guild/index`).

Clone this repository:

```
$ mkdir -p ~/.guild/index
$ git clone git@github.com:guildai/index.git ~/.guild/index
```

To update the index, either use `git` manually:

```
$ git pull -C ~/.guild/index
```

or run Guild update:

```
$ guild update
```

Note that if you install the index in a non-default location, run:

```
$ guild update --index LOCATION
```

where `LOCATION` is the directory of the index.
