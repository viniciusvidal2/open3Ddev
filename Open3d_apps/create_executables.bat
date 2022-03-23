rmdir cloud_executables /S /Q
del cloud_executables.zip

pyinstaller register_space_cap2.py
pyinstaller register_object_cap2.py
rmdir build /S /Q

ren dist cloud_executables
powershell Compress-Archive cloud_executables cloud_executables.zip

rmdir cloud_executables /S /Q
