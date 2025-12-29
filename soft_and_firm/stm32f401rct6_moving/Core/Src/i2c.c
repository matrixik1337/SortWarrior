/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    i2c.c
  * @brief   This file provides code for the configuration
  *          of the I2C instances.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "i2c.h"

/* USER CODE BEGIN 0 */
// Объявляем extern для доступа к переменным из main.c
extern int8_t l_val;
extern int8_t r_val;
extern uint8_t tx_byte;
extern uint8_t rx_buffer[2];
extern volatile uint8_t rx_index;
extern volatile uint8_t i2c_busy;

// Флаг для отладки
volatile uint8_t i2c_event_flag = 0;
/* USER CODE END 0 */

I2C_HandleTypeDef hi2c1;

/* I2C1 init function */
void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 32;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_ENABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_ENABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

void HAL_I2C_MspInit(I2C_HandleTypeDef* i2cHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(i2cHandle->Instance==I2C1)
  {
  /* USER CODE BEGIN I2C1_MspInit 0 */

  /* USER CODE END I2C1_MspInit 0 */

    __HAL_RCC_GPIOB_CLK_ENABLE();
    /**I2C1 GPIO Configuration
    PB6     ------> I2C1_SCL
    PB7     ------> I2C1_SDA
    */
    GPIO_InitStruct.Pin = GPIO_PIN_6|GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /* I2C1 clock enable */
    __HAL_RCC_I2C1_CLK_ENABLE();

    /* I2C1 interrupt Init */
    HAL_NVIC_SetPriority(I2C1_EV_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(I2C1_EV_IRQn);
    HAL_NVIC_SetPriority(I2C1_ER_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(I2C1_ER_IRQn);
  /* USER CODE BEGIN I2C1_MspInit 1 */

  /* USER CODE END I2C1_MspInit 1 */
  }
}

void HAL_I2C_MspDeInit(I2C_HandleTypeDef* i2cHandle)
{

  if(i2cHandle->Instance==I2C1)
  {
  /* USER CODE BEGIN I2C1_MspDeInit 0 */

  /* USER CODE END I2C1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_I2C1_CLK_DISABLE();

    /**I2C1 GPIO Configuration
    PB6     ------> I2C1_SCL
    PB7     ------> I2C1_SDA
    */
    HAL_GPIO_DeInit(GPIOB, GPIO_PIN_6);

    HAL_GPIO_DeInit(GPIOB, GPIO_PIN_7);

    /* I2C1 interrupt Deinit */
    HAL_NVIC_DisableIRQ(I2C1_EV_IRQn);
    HAL_NVIC_DisableIRQ(I2C1_ER_IRQn);
  /* USER CODE BEGIN I2C1_MspDeInit 1 */

  /* USER CODE END I2C1_MspDeInit 1 */
  }
}

/* USER CODE BEGIN 1 */

// ========== I2C CALLBACKS ==========

/**
  * @brief  Обработчик при совпадении адреса
  */
void HAL_I2C_AddrCallback(I2C_HandleTypeDef *hi2c, uint8_t TransferDirection, uint16_t AddrMatchCode)
{
    if(hi2c->Instance == I2C1)
    {
        i2c_busy = 1;
        i2c_event_flag = 1;
        
        if(TransferDirection == I2C_DIRECTION_TRANSMIT)
        {
            // Master хочет читать (команда i2cget)
            // Отправляем один байт
            HAL_I2C_Slave_Transmit_IT(hi2c, &tx_byte, 1);
        }
        else
        {
            // Master хочет писать (команда i2cset)
            rx_index = 0;  // Сбрасываем индекс для нового пакета
        }
    }
}

/**
  * @brief  Обработчик при получении каждого байта
  */
void HAL_I2C_SlaveRxCallback(I2C_HandleTypeDef *hi2c)
{
    if(hi2c->Instance == I2C1)
    {
        // Читаем байт из регистра данных
        uint8_t data = (uint8_t)hi2c->Instance->DR;
        
        if(rx_index < 2)  // Ожидаем максимум 2 байта
        {
            rx_buffer[rx_index] = data;
            rx_index++;
            
            // Если получили 2 байта, обрабатываем сразу
            if(rx_index == 2)
            {
                if(rx_buffer[0] == 1)
                {
                    l_val = (int8_t)(rx_buffer[1] - 100);
                }
                else if(rx_buffer[0] == 2)
                {
                    r_val = (int8_t)(rx_buffer[1] - 100);
                }
            }
        }
    }
}

/**
  * @brief  Обработчик при завершении приёма
  */
void HAL_I2C_SlaveRxCpltCallback(I2C_HandleTypeDef *hi2c)
{
    if(hi2c->Instance == I2C1)
    {
        i2c_busy = 0;
        // Дополнительная обработка если нужно
    }
}

/**
  * @brief  Обработчик при завершении передачи
  */
void HAL_I2C_SlaveTxCpltCallback(I2C_HandleTypeDef *hi2c)
{
    if(hi2c->Instance == I2C1)
    {
        i2c_busy = 0;
    }
}

/**
  * @brief  Обработчик при завершении прослушивания (после STOP)
  */
void HAL_I2C_ListenCpltCallback(I2C_HandleTypeDef *hi2c)
{
    if(hi2c->Instance == I2C1)
    {
        // Очищаем флаги
        __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_AF);
        __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_BERR);
        
        i2c_busy = 0;
        
        // ВАЖНО: Перезапускаем прослушивание
        HAL_I2C_EnableListen_IT(hi2c);
    }
}

/**
  * @brief  Обработчик ошибок I2C
  */
void HAL_I2C_ErrorCallback(I2C_HandleTypeDef *hi2c)
{
    if(hi2c->Instance == I2C1)
    {
        uint32_t error = HAL_I2C_GetError(hi2c);
        
        // Очищаем флаги ошибок
        if(error & HAL_I2C_ERROR_AF)
        {
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_AF);
        }
        if(error & HAL_I2C_ERROR_BERR)
        {
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_BERR);
        }
        if(error & HAL_I2C_ERROR_ARLO)
        {
            __HAL_I2C_CLEAR_FLAG(hi2c, I2C_FLAG_ARLO);
        }
        
        // Сбрасываем состояние
        hi2c->State = HAL_I2C_STATE_READY;
        
        // Перезапускаем прослушивание
        HAL_I2C_EnableListen_IT(hi2c);
    }
}

/* USER CODE END 1 */
