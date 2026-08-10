set CONFIG=Debug

cmake -S . -B "./.build" -A x64  -DCMAKE_BUILD_TYPE=%CONFIG%

pause
exit
