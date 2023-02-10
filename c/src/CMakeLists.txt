cmake_minimum_required(VERSION 3.7)

# add the executable
add_executable(stream.exe stream.c)

# add libm for linking
if(HAVE_SQRT)
  target_link_libraries(stream.exe m)
endif(HAVE_SQRT)

install(TARGETS stream.exe DESTINATION bin)
