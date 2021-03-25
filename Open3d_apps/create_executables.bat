rem Apaga repositorios antigos se existirem
rmdir register_space_cap2 /S /Q
rmdir register_object_cap2 /S /Q
rmdir dist /S /Q
rmdir cloud_executables /S /Q

rem Compila para space e renomeia output
python setup_space.py py2exe
ren dist register_space_cap2

rem Compila para object e renomeia output
python setup_object.py py2exe
ren dist register_object_cap2

rem Cria pasta com as duas saidas e as move para ela
mkdir cloud_executables
move register_object_cap2 cloud_executables
move register_space_cap2 cloud_executables
powershell Compress-Archive cloud_executables cloud_executables.zip

rem Move para o Google Drive
del C:\Users\vinic\GoogleDrive\Executable_CapDesktop\cloud_executables.zip
move cloud_executables.zip C:\Users\vinic\GoogleDrive\Executable_CapDesktop

rem Apaga repositorios antigos se existirem
rmdir register_space_cap2 /S /Q
rmdir register_object_cap2 /S /Q
rmdir dist /S /Q
rmdir cloud_executables /S /Q
