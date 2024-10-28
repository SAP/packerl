#!/bin/bash

# function to display help message
display_help() {
  echo "Usage: $0 [options] [arguments]"
  echo "Options:"
  echo "  -r    Rebuild ns3 (e.g. when configuring for a different mode, or when installing a new version)"
  echo "  -d    If specified, build in default mode with logs. if not, builds in optimized mode"
  echo "  -m    Install the requirements for using the MAGNNETO baseline (i.e. tensorflow)"
  echo "  -h    Display this help message"
}

# initialize variables to store option values
rebuild=false
debug_mode=false
install_magnneto=false

# parse command line options
while getopts ":rdsmlh" opt; do
  case $opt in
    r)
      rebuild=true
      ;;
    d)
      debug_mode=true
      ;;
    m)
      install_magnneto=true
      ;;
    h)
      display_help
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# shift the options out so that $1 refers to the first non-option argument
shift $((OPTIND - 1))

# checkout the correct branch for ns3-ai
ns3aibranch=$(git config -f .gitmodules submodule.ns3-ai.branch)
pushd ns3-ai/
  git switch $ns3aibranch
popd

# install tensorflow if needed
if $install_magnneto; then
  pip install tensorflow==2.15
fi

# remove ns3-packerl scratch if rebuilding
if $rebuild; then
  rm -rf ns3/scratch/ns3-packerl
fi

# copy custom ns3 modules to their place in ns3
cp -r ns3-ai ns3/contrib
cp -r ns3-packerl ns3/scratch

# install ns3-ai and ns3
ns3branch=$(git config -f .gitmodules submodule.ns3.branch)
pushd ns3
  git switch $ns3branch
  pushd contrib/ns3-ai/py_interface/
    pip install -e .
  popd
  if $rebuild; then
    ./ns3 clean
  fi
  if $debug_mode; then
    ./ns3 configure -d default --enable-asserts --enable-logs
  else
    ./ns3 configure -d optimized --enable-asserts
  fi
  ./ns3 build
popd
