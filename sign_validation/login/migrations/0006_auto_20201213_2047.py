# Generated by Django 3.1.2 on 2020-12-13 12:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('login', '0005_auto_20201127_2035'),
    ]

    operations = [
        migrations.RenameField(
            model_name='doubleimages',
            old_name='title1',
            new_name='referenceimg_title',
        ),
        migrations.RenameField(
            model_name='doubleimages',
            old_name='title2',
            new_name='testimg_title',
        ),
        migrations.RemoveField(
            model_name='doubleimages',
            name='image1',
        ),
        migrations.RemoveField(
            model_name='doubleimages',
            name='image2',
        ),
        migrations.AddField(
            model_name='doubleimages',
            name='referenceimg_image',
            field=models.ImageField(null=True, upload_to='doubleimage/referenceimg_image'),
        ),
        migrations.AddField(
            model_name='doubleimages',
            name='testimg_image',
            field=models.ImageField(null=True, upload_to='doubleimage/testimg_image'),
        ),
    ]
