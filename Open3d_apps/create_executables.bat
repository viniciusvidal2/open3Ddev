rem Apaga repositorios antigos se existirem
rmdir space /S /Q
rmdir object /S /Q
rmdir dist /S /Q
rmdir cloud_executables /S /Q
del cloud_executables.zip

rem Compila para space e renomeia output
pyinstaller register_space_cap2.py
ren dist space
rmdir build /S /Q

rem Compila para object e renomeia output
pyinstaller register_object_cap2.py
ren dist object
rmdir build /S /Q

rem Cria pasta com as duas saidas e as move para ela
mkdir cloud_executables
move object/register_object_cap2 cloud_executables
move space/register_space_cap2 cloud_executables
powershell Compress-Archive cloud_executables cloud_executables.zip

rem Apaga repositorios antigos se existirem
rmdir object/S /Q
rmdir space/S /Q
rmdir dist /S /Q
rmdir cloud_executables /S /Q
