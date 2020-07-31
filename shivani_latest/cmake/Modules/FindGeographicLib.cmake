# Look for GeographicLib
#
# Set
#  GeographicLib_FOUND = GEOGRAPHICLIB_FOUND = TRUE
#  GeographicLib_INCLUDE_DIRS = /usr/local/include
#  GeographicLib_LIBRARIES = /usr/local/lib/libGeographic.so
#  GeographicLib_LIBRARY_DIRS = /usr/local/lib

# Use GEOGRAPHICLIB as root, if defined - otherwise use mine (bpd6-ASAP):
if( DEFINED ENV{GEOGRAPHICLIB} )
  set( GEOGRAPHICLIB_ROOT $ENV{GEOGRAPHICLIB} )
endif()

if ( NOT DEFINED GEOGRAPHICLIB_ROOT AND NOT DEFINED PKG_GEOGRAPHIC )
        set ( GEOGRAPHICLIB_ROOT "/glade/u/home/bdobbins/Software/GeographicLib" )
endif()

#Check whether to search static or dynamic libs
set( CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES} )

find_library ( GeographicLib_LIBRARIES
  NAMES Geographic
  HINTS "${GEOGRAPHICLIB_ROOT}/lib"
) 

if (GeographicLib_LIBRARIES)
  get_filename_component (GeographicLib_LIBRARY_DIRS  "${GeographicLib_LIBRARIES}" PATH)
  get_filename_component (_ROOT_DIR "${GeographicLib_LIBRARY_DIRS}" PATH)
  set (GeographicLib_INCLUDE_DIRS "${_ROOT_DIR}/include")
  set (GeographicLib_BINARY_DIRS "${_ROOT_DIR}/bin")
  if (NOT EXISTS "${GeographicLib_INCLUDE_DIRS}/GeographicLib/Config.h")
    set ( INSIDE "true" )
    # On Debian systems the library is in e.g.,
    #   /usr/lib/x86_64-linux-gnu/libGeographic.so
    # so try stripping another element off _ROOT_DIR
    get_filename_component (_ROOT_DIR "${_ROOT_DIR}" PATH)
    set (GeographicLib_INCLUDE_DIRS "${_ROOT_DIR}/include")
    set (GeographicLib_BINARY_DIRS "${_ROOT_DIR}/bin")
    if (NOT EXISTS "${GeographicLib_INCLUDE_DIRS}/GeographicLib/Config.h")
      unset (GeographicLib_INCLUDE_DIRS)
      unset (GeographicLib_LIBRARIES)
      unset (GeographicLib_LIBRARY_DIRS)
      unset (GeographicLib_BINARY_DIRS)
    endif ()
  endif ()
  unset (_ROOT_DIR)
  set ( GEOFOUND "true" )
  set ( INSIDE "sure" )
else()
  set ( GEOFOUND "false" )
endif()

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (GeographicLib DEFAULT_MSG
  GeographicLib_LIBRARY_DIRS GeographicLib_LIBRARIES
  GeographicLib_INCLUDE_DIRS)
mark_as_advanced (GeographicLib_LIBRARY_DIRS GeographicLib_LIBRARIES
  GeographicLib_INCLUDE_DIRS)
