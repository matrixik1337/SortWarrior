/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define LED_Pin GPIO_PIN_13
#define LED_GPIO_Port GPIOC
#define MLENCA_Pin GPIO_PIN_0
#define MLENCA_GPIO_Port GPIOC
#define MLENCB_Pin GPIO_PIN_1
#define MLENCB_GPIO_Port GPIOC
#define MRENCA_Pin GPIO_PIN_2
#define MRENCA_GPIO_Port GPIOC
#define MRENCB_Pin GPIO_PIN_3
#define MRENCB_GPIO_Port GPIOC
#define MLPWM_Pin GPIO_PIN_0
#define MLPWM_GPIO_Port GPIOA
#define MRPWM_Pin GPIO_PIN_1
#define MRPWM_GPIO_Port GPIOA
#define ML1_Pin GPIO_PIN_2
#define ML1_GPIO_Port GPIOA
#define MR1_Pin GPIO_PIN_3
#define MR1_GPIO_Port GPIOA
#define ML2_Pin GPIO_PIN_4
#define ML2_GPIO_Port GPIOA
#define MR2_Pin GPIO_PIN_5
#define MR2_GPIO_Port GPIOA

/* USER CODE BEGIN Private defines */

static uint8_t uart_rx_buffer[2];

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
