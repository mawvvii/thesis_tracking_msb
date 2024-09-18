#!/bin/sh

# $1 = final shared object file to be loaded (doesn't need to be installed)
# $2 = Name of extension
# $3 = Version of extension

# for non-release versions, convert the '~dev.' portion into something compliant
# with TCL’s version number rules
version=$(echo "$3" | sed 's/~dev\./b/g')

echo "package ifneeded $2 ${version} \"" >pkgIndex.tcl
case "$1" in
  *tk* )
    echo "	package require Tk 8.3" >>pkgIndex.tcl
    ;;
esac
echo "	load [file join \$dir $1] $2\"" >>pkgIndex.tcl
