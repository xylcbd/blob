### php不加载插件
直接终端执行php test.php可以，但是在php web和crontab中执行则出现undefined function。
重启php-fpm服务：
sudo /etc/init.d/php-fpm restart
无法加载opencv动态库，修改LDD环境变量：
sudo vim /etc/ld.so.conf.d/custom.conf，增加/usr/local/lib