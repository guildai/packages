# Guild AI Packages

This is the active source repository for Guild AI managed packages.

Each sub-directory represents a package. Refer to each package
`README.md` for details.

Packages are built and deployed by members of the Guild AI package
team. They're available for installation using the Guild AI command
line application. See [Installing packages](#installing-packages)
below for details.

Links:

- **[Package overview](https://www.guild.ai/docs/package/)**
- **[Packaging developers guide](https://www.guild.ai/docs/developer/packaging/)**
- **[Open issues](https://github.com/guildai/packages/issues)**

## Installing packages

To use a package, you must install it first using `guild install`.

Ensure that you have the latest version of Guild AI:

```
$ pip install guildai --upgrade
```

Verify your installation:

```
$ guild check
```

If you know the name of the package you want to install, install it by
running:

```
$ guild install PACKAGE
```

For example, to install the `slim.resnet` package, run:

```
$ guild install slim.resnet
```

### Finding packages

To find a package, browse this repository or use `guild search`:

```
$ guild search TERM
```

### Getting package information

Once a package is installed, you can read more about it by running:

```
$ guild help PACKAGE
```

To see a list of models available in the package, run:

```
$ guild models PACKAGE
```

To see a list of operations available in the package, run:

```
$ guild operations PACKAGE
```

### Running package model operations

Packages are primarily used to distribute models. You can run a model
operation in Guild AI this way:

```
$ guild run MODEL:OPERATION [FLAG...]
```

`MODEL` is one of the models in the installed package and `OPERATION`
is the model operation you want to run.

To get help on a particular operation, run:

```
$ guild run MODEL:OPERATION --help-op
```
