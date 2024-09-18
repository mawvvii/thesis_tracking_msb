FROM fedora:40

COPY requirements.txt ./requirements.txt

# Development tools
RUN dnf -y --setopt install_weak_deps=false groupinstall 'Development Tools' \
&& dnf -y --setopt install_weak_deps=false install rpm-build \
bison \
byacc \
flex \
# Centos build utilities
libtool-ltdl-devel \
libedit-devel \
# Option xpat
expat-devel \
# Option glut
freeglut-devel \
# Option ann
ann-devel \
# Option gts
gts-devel \
# Option swig
swig \
# Command smyra
# Command gvedit
qt5-qtbase-devel \
ncurses-devel \
# Plugin library devil
DevIL-devel \
# Plugin library gd
gd-devel \
# Documentation and Plugin library ghostscipt
ghostscript ghostscript-devel \
# Plugin library gdk
gtk3-devel \
# Plugin library gtk
gtkglext-devel \
# Plugin library lasi
lasi-devel \
# Plugin library poppler
poppler-glib-devel \
poppler-qt5 \
# Plugin library rsvg
librsvg2-devel \
# Language extension gv_go
golang \
gcc-go \
# Language extension gv_guile
guile30-devel \
# Language extension gv_java
java-devel \
# Language extension gv_lua
lua-devel \
# Language extension gv_perl
perl \
perl-ExtUtils-Embed \
# Language extension gv_php
php-devel \
# Language extension gv_python3
python3-devel \
# Language extension gv_R
R \
R-devel \
# Language extension gv_ruby
ruby-devel \
# Language extension gv_tcl
# Required in redhat/graphviz.spec.fedora.in
libglade2-devel \
mono-core \
# create a Python virtual environment
&& python3 -m venv /opt/virtualenv \
# install Python dependencies within the virtual environment
&& env PATH="/opt/virtualenv/bin:$PATH" python3 -m pip install --requirement requirements.txt \
# install a newer CMake to deal with RPM 4.19 glob changes
&& curl --retry 3 --location --no-progress-meter -O \
  https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.tar.gz \
&& tar xf cmake-3.28.1-linux-x86_64.tar.gz \
&& mv cmake-3.28.1-linux-x86_64/bin/* /usr/bin/ \
&& mv cmake-3.28.1-linux-x86_64/share/cmake-3.28 /usr/share/ \
&& rm -rf cmake-3.28.1-linux-x86_64.tar.gz cmake-3.28.1-linux-x86_64 \
# Clean up
&& dnf clean all && rm -rf /var/cache/dnf
